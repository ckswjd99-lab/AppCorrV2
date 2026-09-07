"""
block.py

Forked timm `Block` (see timm/models/vision_transformer.py) exposing `.approx()` / `.correct()`,
mirroring `appcorr/models/dinov3/layers/block.py::SelfAttentionBlock.approx_partial_token/
correct_partial_token` but dropping rope (not used by these towers) and the token-importance-based
sub-pruning DINOv3 layers on top (`token_keep_ratio`/pscore machinery) -- our v1 takes the corrected
token set (`token_idx`) as given from the caller (Phase 3's executor decides which patches got new
data this round), rather than further pruning it internally. That refinement can be ported later
once the base mechanism is validated.

Key invariants (audited against DINOv3 and validated empirically -- see
analysis/experiments/audit_progressive_semantics.py):
    - `x` is the full [B, N, C] residual stream threaded through every block's `.correct()` call;
      each correction round restarts it from layer-0 tokens (same as DINOv3's executor, which sets
      `x_temp = context['input_tokens']` per CORRECT_FORWARD).
    - Positions in `token_idx` (same set across layers within one round) hold freshly recomputed
      values. Non-queried positions are reconstructed as `x + {tag}_blocks_out_sum` (the approx
      pass's cached block delta). These reconstructed values equal the approx pass's stream only if
      the layer-0 tokens fed to this round match the approx pass at non-corrected positions (true in
      AppCorr, whose canvas keeps non-corrected regions at base-layer resolution). Either way they
      are DEAD VALUES: attention reads keys/values from the K/V cache (never from the stream), and
      norm1/MLP touch only `token_idx` rows -- so nothing consumed downstream depends on them
      (empirically confirmed: full-true-image vs faithful-partial-canvas inputs give bit-identical
      last-position logits at partial correction).
    - Multi-round correction is NOT bit-exact for this bidirectional tower, regardless of ordering:
      a round-1 token's K/V at layers >= 1 is computed while later-round tokens are still stale, and
      is never revisited. Empirically: single-round 100% correction matches stock to bf16 kernel
      noise (max last-logit err ~0.5), while two-round sequential 100% leaves ~4.1 max logit error
      (though the decoded action bins typically still agree). This staleness is inherent to the
      AppCorr accepted-approximation design (DINOv3 multi-group correction has the same property),
      not a bug -- but do not claim bit-exact convergence for any multi-round schedule.
"""

from typing import Any, Dict

import torch
import torch.nn as nn

from .attention import ApproxCorrectAttention


class ApproxCorrectBlock(nn.Module):
    def __init__(self, norm1: nn.Module, attn: ApproxCorrectAttention, ls1: nn.Module,
                 norm2: nn.Module, mlp: nn.Module, ls2: nn.Module):
        super().__init__()
        self.norm1 = norm1
        self.attn = attn
        self.ls1 = ls1
        self.norm2 = norm2
        self.mlp = mlp
        self.ls2 = ls2

    @classmethod
    def from_stock(cls, blk: nn.Module) -> "ApproxCorrectBlock":
        return cls(
            norm1=blk.norm1,
            attn=ApproxCorrectAttention.from_stock(blk.attn),
            ls1=blk.ls1,
            norm2=blk.norm2,
            mlp=blk.mlp,
            ls2=blk.ls2,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Identical to stock `Block.forward` (drop_path is nn.Identity() in eval mode for these
        towers, so it is omitted rather than threaded through -- see timm Block.forward)."""
        x = x + self.ls1(self.attn(self.norm1(x)))
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x

    def approx(self, x: torch.Tensor, cache_feature: Dict[str, Any], tag: str,
               collect_cls_attn: bool = False):
        """Full block forward over all N tokens; caches the total block delta (`{tag}_blocks_out_sum`
        = attn contribution + FFN contribution) so `.correct()` can reconstruct stale positions
        exactly via `x_in + blocks_out_sum`, and caches raw K/V (via `self.attn.approx`) so
        `.correct()` can splice in fresh K/V for corrected positions."""
        x_attn, cache_feature = self.attn.approx(self.norm1(x), cache_feature, tag,
                                                 collect_cls_attn=collect_cls_attn)
        x_attn_out = self.ls1(x_attn)
        cache_feature[f"{tag}_blocks_out_sum"] = x_attn_out.detach().clone()

        x_mid = x + x_attn_out
        mlp_out = self.ls2(self.mlp(self.norm2(x_mid)))
        cache_feature[f"{tag}_blocks_out_sum"] = cache_feature[f"{tag}_blocks_out_sum"] + mlp_out.detach()

        x_out = x_mid + mlp_out
        return x_out, cache_feature

    def correct(self, x: torch.Tensor, token_idx: torch.Tensor, cache_feature: Dict[str, Any], tag: str):
        """
        Args:
            x: [B, N, C] -- current residual stream (see module docstring for the staleness invariant).
            token_idx: [Q] -- absolute positions being corrected this round (same set across layers
                within one correction round; the caller may use a different set on the next round).
        Returns:
            x_out: [B, N, C] -- `token_idx` positions hold freshly recomputed values; all other
                positions are reconstructed to exactly match a full approx-only forward.
        """
        token_idx = token_idx.to(x.device)
        x_active = x[:, token_idx]  # [B, Q, C]
        x_norm_sel = self.norm1(x_active)

        x_attn_sel, cache_feature = self.attn.correct(x_norm_sel, token_idx, cache_feature, tag)
        x_attn_active = x_active + self.ls1(x_attn_sel)
        mlp_out_new = self.ls2(self.mlp(self.norm2(x_attn_active)))

        blocks_out_sum = cache_feature[f"{tag}_blocks_out_sum"]
        x_out = x + blocks_out_sum.to(dtype=x.dtype)
        x_out = x_out.clone()
        x_out[:, token_idx] = (x_attn_active + mlp_out_new).to(dtype=x_out.dtype)

        # Persist this block's *corrected* increment over the approximate one (DINOv3
        # `SelfAttentionBlock.correct_partial_token`, ac0238f): a later round that replays this
        # block for a different token set rebuilds these rows as `x + blocks_out_sum`, and without
        # the write it rebuilds them from the stale approx increment -- every earlier round's
        # correction of a row survives only through the K/V cache, its own value is discarded.
        # `ls1(x_attn_sel) + mlp_out_new` is exactly what `.approx()` would have stored, so the two
        # paths stay interchangeable. No-op under cumulative correction (every corrected row is in
        # the query set again next round, so the stored value is never read back); it is what makes
        # a new-only schedule (each round corrects only its own group) accumulate at all.
        blocks_out_sum[:, token_idx] = ((x_attn_active - x_active) + mlp_out_new).to(blocks_out_sum.dtype)

        return x_out, cache_feature
