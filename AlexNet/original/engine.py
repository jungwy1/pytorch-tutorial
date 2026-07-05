import torch

# ============================================================================
# Training / evaluation engine
# ----------------------------------------------------------------------------
# Model-agnostic loops (same code works for LeNet, AlexNet, ...). The loss and
# optimizer are passed in, so this file never mentions a specific network.
#
# Core training step (repeated per batch):
#   pred = model(X)          forward pass  -> logits
#   loss = loss_fn(pred, y)  compare to labels
#   loss.backward()          backprop      -> gradients in each param's .grad
#   optimizer.step()         update weights from .grad
#   optimizer.zero_grad()    clear .grad (PyTorch ACCUMULATES grads by default)
#
# train() vs eval():
#   model.train() turns Dropout ON  (used while learning).
#   model.eval()  turns Dropout OFF (used for evaluation).
#   Matters for AlexNet because it has Dropout in the FC layers.
#
# Averaging rule in test_loop:
#   test_loss is accumulated per BATCH (loss_fn returns a batch-mean scalar),
#     so divide by num_batches = len(dataloader).
#   correct is accumulated per SAMPLE (one hit per image),
#     so divide by size = len(dataloader.dataset).
#
# torch.no_grad() in test_loop: disables autograd (no graph is built) -> less
#   memory and faster, since we never call backward() during evaluation.
# ============================================================================


# One pass over the training set: forward -> loss -> backward -> update.
def train_loop(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)      # total number of training samples
    model.train()                       # train mode: dropout ON
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)   # move this batch to GPU/CPU

        pred = model(X)                 # forward -> logits (N, num_classes)
        loss = loss_fn(pred, y)         # CrossEntropyLoss applies softmax inside

        loss.backward()                 # backprop: fill .grad on every parameter
        optimizer.step()                # update weights using .grad
        optimizer.zero_grad()           # reset grads (PyTorch accumulates them)

        if batch % 20 == 0:             # periodic progress log
            # .item() detaches the scalar from the graph -> plain float
            loss, current = loss.item(), batch * dataloader.batch_size + len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")


# One pass over the val set: no gradients, report accuracy + average loss.
def test_loop(dataloader, model, loss_fn, device):
    model.eval()                        # eval mode: dropout OFF
    size = len(dataloader.dataset)      # total samples (for accuracy)
    num_batches = len(dataloader)       # number of batches (for avg loss)
    test_loss, correct = 0, 0
    with torch.no_grad():               # no graph/grad -> less memory, faster
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            # loss_fn returns the batch-mean loss (a scalar) -> accumulate it
            test_loss += loss_fn(pred, y).item()
            # argmax(1) = predicted class per sample; compare to labels, count hits
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches            # batch-accumulated -> divide by #batches
    correct /= size                     # sample-accumulated -> divide by #samples
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
