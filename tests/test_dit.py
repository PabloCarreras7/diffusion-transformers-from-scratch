import unittest
from DiT import Patchify, SelfAttention, MultiHeadSelfAttn, FeedForward, DiTBlock, Unpatchify, DiT
import torch


res = 128
patch_size = 16
channels = 3
heads = 8
batch_size = 4
patch_dim = 768
num_patch = 64
ff_dim = 2048
time_emb_dim = 512
num_blocks = 5
num_timesteps = 1000
hidden_dim = 768


'''
Test Bench:
    * Tests all of the modules in the DiT model
    * Note these tests are not exhaustive, you may create more tests in another file.
    * To run: python tests.py.
    * DO NOT CHANGE THESE TESTS!
'''


class TestBench(unittest.TestCase):
    
    
    ## Test Patchify
    def test_patchify(self):
        P = Patchify(patch_size)
        x = torch.ones((batch_size, channels, res, res))
        expected = torch.ones((batch_size, num_patch, patch_dim))
        out = P(x)
        self.assertTrue(out.shape == expected.shape)
        self.assertTrue(torch.equal(out, expected))


    ## Test Unpatchify
    def testUnpatchify(self):
        x = torch.ones((batch_size, num_patch, hidden_dim))
        P = Unpatchify(patch_size)
        out = P(x)
        expected = torch.ones((batch_size, channels, res, res))
        self.assertTrue(torch.equal(out, expected))

    ## Patch and Unpatchify
    def testPatchAndUnpatchify(self):
        x = torch.randn((batch_size, channels, res, res))
        P = Patchify(patch_size)
        out = P(x)
        expected = torch.randn((batch_size, num_patch, patch_dim))
        self.assertTrue(out.shape == expected.shape)
        U = Unpatchify(patch_size)
        out = U(out)
        self.assertTrue(torch.equal(out, x))

    ## Test SelfAttention
    def testSelfAttn(self):
        inner_dim = hidden_dim // 2
        selfAttn = SelfAttention(hidden_dim, inner_dim)
        res = torch.ones((batch_size, num_patch, inner_dim))
        x = torch.randn(batch_size, num_patch, hidden_dim)
        out = selfAttn(x)
        self.assertTrue(out.shape == res.shape)

    ## Test MultiHeadSelfAttention
    def testMultiHead(self):
        multiHead = MultiHeadSelfAttn(hidden_dim, heads)
        x = torch.rand((batch_size, num_patch, hidden_dim))
        out = multiHead(x)
        self.assertTrue(x.shape == out.shape)


    ## Test FeedForward
    def testFF(self):
        x = torch.rand((batch_size, num_patch, hidden_dim))
        FF = FeedForward(hidden_dim, ff_dim)
        out = FF(x)
        self.assertTrue(x.shape == out.shape)

    ## Test DiTBlock
    def testDiTBlock(self):
        block = DiTBlock(hidden_dim=hidden_dim, num_heads=heads, ff_dim=ff_dim, time_emb_dim=time_emb_dim)
        x = torch.rand((batch_size, num_patch, hidden_dim))
        t = torch.rand((batch_size, time_emb_dim))
        out = block(x, t)
        
        self.assertTrue(out.shape == x.shape)
        self.assertTrue(torch.equal(out, x))

    
    ## Test DiT
    def testDiT(self):
        x = torch.rand((batch_size, channels, res, res))
        t = torch.randint(low=0, high=num_timesteps-1, size=(batch_size, ))
        model = DiT(patch_size=patch_size, num_blocks=num_blocks, num_heads=heads, ff_dim=ff_dim, num_patches=num_patch,
        num_timesteps=num_timesteps, time_emb_dim=time_emb_dim, num_channels=channels, hidden_dim=hidden_dim)
        out = model(x, t)
        
        self.assertTrue(out.shape == x.shape)

    ## Test DiTBlock is a residual connection (zero-initialized adaLN outputs identity)
    def testDiTBlockResidual(self):
        block = DiTBlock(hidden_dim=hidden_dim, num_heads=heads, ff_dim=ff_dim, time_emb_dim=time_emb_dim)
        x = torch.rand((batch_size, num_patch, hidden_dim))
        t1 = torch.rand((batch_size, time_emb_dim))
        t2 = torch.rand((batch_size, time_emb_dim))
        out1 = block(x, t1)
        out2 = block(x, t2)
        self.assertTrue(torch.equal(out1, out2))

    ## Test DiT with different resolutions and patch sizes
    def testDiTDifferentConfig(self):
        alt_res = 64
        alt_patch_size = 8
        alt_channels = 1
        alt_num_patch = 64
        x = torch.rand((batch_size, alt_channels, alt_res, alt_res))
        t = torch.randint(low=0, high=num_timesteps - 1, size=(batch_size,))
        model = DiT(patch_size=alt_patch_size, num_blocks=num_blocks, num_heads=heads, ff_dim=ff_dim, num_patches=alt_num_patch,
            num_timesteps=num_timesteps, time_emb_dim=time_emb_dim, num_channels=alt_channels, hidden_dim=hidden_dim)
        out = model(x, t)
        self.assertTrue(out.shape == x.shape)


if __name__ == "__main__":
    unittest.main()
    