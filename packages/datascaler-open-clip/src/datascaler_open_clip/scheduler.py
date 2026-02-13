import math

from torch.optim.lr_scheduler import LambdaLR


def cosine_lr_original(optimizer, base_lr, warmup_length, steps):
    def _lr_adjuster(step):
        if step < warmup_length:
            lr = base_lr * (step + 1) / warmup_length
        else:
            e = step - warmup_length
            es = steps - warmup_length
            lr = 0.5 * (1 + math.cos(math.pi * e / es)) * base_lr

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        return lr

    return _lr_adjuster


def cosine_lr(optimizer, warmup_length, steps):
    def lr_lambda(step):
        if step < warmup_length:
            return step / warmup_length

        e = step - warmup_length
        es = steps - warmup_length
        return 0.5 * (1.0 + math.cos(math.pi * e / es))

    return LambdaLR(optimizer, lr_lambda)


def cosine_lr_with_end(optimizer, warmup_length, steps, end_lr=1e-5):
    base_lrs = [group["lr"] for group in optimizer.param_groups]

    def make_lr_lambda(base_lr):
        def lr_lambda(step):
            if step < warmup_length:
                return step / warmup_length

            e = step - warmup_length
            es = steps - warmup_length
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * e / es))
            scaled_lr = end_lr + (base_lr - end_lr) * cosine_decay
            return scaled_lr / base_lr

        return lr_lambda

    lambdas = [make_lr_lambda(base_lr) for base_lr in base_lrs]
    return LambdaLR(optimizer, lambdas)


def linear_lr(optimizer, warmup_length, steps):
    def lr_lambda(step):
        if step < warmup_length:
            return step / warmup_length

        e = step - warmup_length
        es = steps - warmup_length
        return max(0.0, 1.0 - e / es)

    return LambdaLR(optimizer, lr_lambda)


def linear_lr_with_end(optimizer, warmup_length, steps, end_lr=1e-5):
    base_lrs = [group["lr"] for group in optimizer.param_groups]

    def make_lr_lambda(base_lr):
        def lr_lambda(step):
            if step < warmup_length:
                return step / warmup_length

            e = step - warmup_length
            es = steps - warmup_length
            scaled_lr = end_lr + (base_lr - end_lr) * (1.0 - e / es)
            return scaled_lr / base_lr

        return lr_lambda

    lambdas = [make_lr_lambda(base_lr) for base_lr in base_lrs]
    return LambdaLR(optimizer, lambdas)
