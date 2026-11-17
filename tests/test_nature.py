import torch

from week03_attention.CausalSelfAttention import CausalSelfAttention

"""
测试batch元素相互独立, 修改第二个样本不影响第一个样本的输出
1. 创建一个batch包含两个样本
2. 创建注意力模型, 调用eval(), 计算修改前的输出
3. 修改第二个样本
4. 使用同一个模型再次计算修改后的输出
5. 比较第一个样本的两次输出是否相同
"""


def test_batch_isolate():
    torch.manual_seed(0)
    batchs = torch.randn(2, 4, 8)
    n_embd = 8
    n_head = 2
    block_size = 10
    model = CausalSelfAttention(n_embd, n_head, block_size)
    with torch.no_grad():
        model.eval()
        output1 = model(batchs)
        batchs[1] = torch.randn_like(batchs[0])
        output2 = model(batchs)
    torch.testing.assert_close(output1[0], output2[0])


"""
测试因果注意力模型中经过softmax之后行和是不是等于1
"""


def test_softmax_sum():
    torch.manual_seed(0)
    batchs = torch.randn(2, 4, 8)
    option_test = True
    n_embd = 8
    n_head = 2
    block_size = 10
    model = CausalSelfAttention(n_embd, n_head, block_size)
    with torch.no_grad():
        model.eval()
        _, wei = model(batchs, option_test)
    assert wei.shape == (2, 2, 4, 4)
    row_sums = wei.sum(dim=-1)
    torch.testing.assert_close(row_sums, torch.ones_like(row_sums))


"""
测试mask上三角是不是全部为0
1. 取出wei的全部上三角元素
2. 对比这些元素是不是全部为0
"""


def test_mask():
    torch.manual_seed(0)
    batchs = torch.randn(2, 4, 8)
    option_test = True
    n_embd = 8
    n_head = 2
    block_size = 10
    model = CausalSelfAttention(n_embd, n_head, block_size)
    with torch.no_grad():
        model.eval()
        _, wei = model(batchs, option_test)
    upper_mask = torch.ones_like(wei, dtype=torch.bool).triu(diagonal=1)
    upper_values = wei[upper_mask]
    assert torch.all(upper_values == 0)


"""
测试eval的确定性
"""


def test_eval():
    torch.manual_seed(0)
    batchs = torch.randn(2, 4, 8)
    n_embd = 8
    n_head = 2
    block_size = 10
    model = CausalSelfAttention(n_embd, n_head, block_size, dropout=0.5)
    with torch.no_grad():
        model.eval()
        out1 = model(batchs)
        out2 = model(batchs)
    torch.testing.assert_close(out1, out2)
