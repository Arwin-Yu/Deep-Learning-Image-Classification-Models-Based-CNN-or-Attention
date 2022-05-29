import torch 

def warmup(optimizer, warm_up_iters, warm_up_factor):
    def f(x):
        """根据step数返回一个学习率倍率因子, x代表step"""
        if x >= warm_up_iters:
            return 1
        
        alpha = float(x) / warm_up_iters
        # 迭代过程中倍率因子从warmup_factor -> 1
        return warm_up_factor * (1 - alpha) + alpha
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)