import sys
import os
import torch
from torch.utils.data import DataLoader

# load the server_setup/fun_code/genAI/llm/lib/utils
sys.path.append(os.path.expanduser("~/server_setup/fun_code/genAI/llm/lib"))

from utils import print_openai_stream, repl
from utils import ChatVisionBot


def print_debug(f):
    def _f(*args, **kwargs):
        o = f(*args, **kwargs)
        print(f"output of {f} is {o}")
        return o

    return _f


class LLM:
    """abstract LLM class"""

    def __init__(self, system_prompt: str):
        self.system_prompt = system_prompt
        self.chatbot = ChatVisionBot(
            system_prompt,
            use_azure=True,
            stream=False,
        )

    def __call__(self, query: str) -> str:
        """query the model with the given prompt"""
        return self.chatbot(query)


class Model:
    def __init__(self, theta: str, d_in: int, d_out: int):
        # theta here is part of the system prompt
        self.theta = theta
        self.d_in = d_in
        self.d_out = d_out
        self.llm = LLM(theta + f"input_dim={d_in}, output_dim={d_out}")

    def set_theta(self, theta: str):
        """set the model parameters"""
        self.theta = theta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward pass"""
        return self.llm(f"forward(x={x})")

    @print_debug
    def __call__(self, x):
        return self.forward(x)


class Criterion:
    def __init__(self, theta: str):
        self.theta = theta
        self.llm = LLM(theta)

    @print_debug
    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """compute the loss"""
        return self.llm(f"loss(y_pred={y_pred}, y_true={y_true})")


class Optimizer:
    def __init__(self, theta: str, model: Model):
        self.theta = theta
        self.model = model
        self.llm = LLM(theta)
        self.zero_grad()

    def zero_grad(self):
        self.examples = []  # inputs
        self.feedbacks = []  # outputs from criterion

    def add_example(
        self,
        x: torch.Tensor,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        loss: torch.Tensor,
    ):
        """add an example to the optimizer"""
        self.examples.append(x)
        self.feedbacks.append(
            self.llm(f"feedback(y_pred={y_pred}, y_true={y_true}, loss={loss})")
        )

    def step(self):
        """update the model parameters"""
        new_theta = self.llm(
            f"update(theta={self.model.theta}, examples={self.examples}, feedbacks={self.feedbacks})"
        )
        self.model.set_theta(new_theta)


# Dummy dataset with random numbers
class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, n_samples=100, d_in=10, d_out=2):
        # Initialize your data here
        self.data = torch.randn(n_samples, d_in)
        self.labels = torch.randint(0, d_out, (n_samples,))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


d_in, d_out = 10, 2
batch_size = 4
n_samples = 8
dataset = SimpleDataset(n_samples, d_in, d_out)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = Model("binary classification model", d_in, d_out)
criterion = Criterion("binary classification loss")
optimizer = Optimizer("adam optimizer", model)

for i, (x, y) in enumerate(dataloader):
    print(f"batch [{i+1} / {len(dataloader)}]")
    optimizer.zero_grad()
    y_pred = model(x)
    loss = criterion(y_pred, y)
    optimizer.add_example(x, y_pred, y, loss)
    optimizer.step()
