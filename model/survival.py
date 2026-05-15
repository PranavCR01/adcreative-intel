import torch
import torch.nn as nn


class WeibullNLLLoss(nn.Module):
    """
    Negative log-likelihood loss for Weibull survival distribution.
    Handles right-censored observations.

    weibull_params: (batch, 2) — log_scale, log_shape
    halflife_days:  (batch,)   — observed time (1.0 for censored rows)
    censored:       (batch,)   — bool, True = right-censored
    """
    def forward(self, weibull_params, halflife_days, censored):
        log_scale, log_shape = weibull_params[:, 0], weibull_params[:, 1]

        # Clamp to prevent NaN — critical, do not remove
        log_scale = torch.clamp(log_scale, -10, 10)
        log_shape = torch.clamp(log_shape, -10, 10)

        scale = torch.exp(log_scale)  # λ (lambda)
        shape = torch.exp(log_shape)  # k

        # Replace null halflife with 1.0 for censored rows (won't affect loss)
        t = torch.clamp(halflife_days, min=1e-6)

        # Log-likelihood: uncensored = log PDF, censored = log survival function
        log_pdf = (log_shape + (shape - 1) * torch.log(t)
                   - shape * log_scale - (t / scale) ** shape)
        log_sf = -((t / scale) ** shape)

        loss = torch.where(censored, -log_sf, -log_pdf)
        return loss.mean()
