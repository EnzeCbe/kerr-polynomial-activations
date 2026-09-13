import torch
import torch.nn as nn
from torch.autograd import gradcheck
import matplotlib.pyplot as plt


def mW_to_W(x):
    return x * 1e-3


def W_to_mW(x):
    return x * 1e3


class OpticalModel(nn.Module):
    def __init__(self, sign=-1.0):
        super().__init__()
        self.log_mag2 = nn.Parameter(torch.randn(1) * 0.1)  # |alpha2|, [W^-1]
        self.log_mag3 = nn.Parameter(torch.randn(1) * 0.1)  # |alpha3|, [W^-2]
        self.sign = sign

    def forward(self, OpticalPowerIn):  # OpticalPowerIn: [W]
        alpha2 = self.sign * torch.exp(self.log_mag2)  # [W^-1]
        alpha3 = self.sign * torch.exp(self.log_mag3)  # [W^-2]
        OpticalPowerOut = OpticalPowerIn * (1 + alpha2 * OpticalPowerIn + alpha3 * OpticalPowerIn**2)  # [W]
        return OpticalPowerOut


model = OpticalModel(sign=-1.0).double()

P_in_test_mW = torch.rand(4, dtype=torch.float64)
P_in_test_W = mW_to_W(P_in_test_mW)
P_in_test_W.requires_grad = True

test_passed = gradcheck(model, (P_in_test_W,))
print("Gradcheck passed:", test_passed)


model = OpticalModel(sign=-1.0)
optimizer = torch.optim.Adam(model.parameters(), lr=0.037)
n_epochs = 100

P_in_mW = torch.linspace(0, 10, 200).unsqueeze(1)  # [mW]
P_in_W = mW_to_W(P_in_mW)  # [W]
P_target_W = mW_to_W(torch.sqrt(P_in_mW))  # target: sqrt(P_in), [W]

loss_history = []

for epoch in range(n_epochs):
    optimizer.zero_grad()
    P_out_W = model(P_in_W)
    loss = torch.mean((P_out_W - P_target_W) ** 2)
    loss.backward()
    optimizer.step()
    loss_history.append(loss.item())

print("Final alpha2 [W^-1]:", (model.sign * torch.exp(model.log_mag2)).item())
print("Final alpha3 [W^-2]:", (model.sign * torch.exp(model.log_mag3)).item())
print("Final loss:", loss_history[-1])

with torch.no_grad():
    P_out_final_W = model(P_in_W)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(P_in_mW.numpy(), W_to_mW(P_target_W).numpy(), label="target: sqrt(P_in)")
axes[0].plot(P_in_mW.numpy(), W_to_mW(P_out_final_W).numpy(), label="learned model")
axes[0].set_xlabel("Optical Power In [mW]")
axes[0].set_ylabel("Optical Power Out [mW]")
axes[0].set_title("Learned activation vs target")
axes[0].legend()

axes[1].plot(range(1, n_epochs + 1), loss_history)
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss (MSE, W^2)")
axes[1].set_title("Training curve")

plt.tight_layout()
plt.savefig("optical_activation_plot.png", dpi=150)
plt.show()
