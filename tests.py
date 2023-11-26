from model import *
import unittest
    
class TestPatchEmbedding(unittest.TestCase):
    def test_forward(self):
        x = torch.randn(2, 3, 256, 256)
        pe = PatchEmbedding()
        y = pe(x)
        self.assertEqual(y.shape, (2, 256, 768))

unittest.main(argv=[''], verbosity=2, exit=False)

class TestTransformerEncoder(unittest.TestCase):
    def test_forward(self):
        x = torch.randn(2, 256, 768)
        te = TransformerEncoder(768, 12, 12, 0.1)
        y = te(x)
        for i in range(len(y)):
            self.assertEqual(y[i].shape, (2, 256, 768))

class TestUNETREncoder(unittest.TestCase):
    def test_forward(self):
        x = torch.randn(2, 3, 256, 256)
        unetr = UNETREncoder()
        y = unetr(x)
        z3, z6, z9, z12 = y
        self.assertEqual(z3.shape, (2, 256, 768))
        self.assertEqual(z6.shape, (2, 256, 768))
        self.assertEqual(z9.shape, (2, 256, 768))
        self.assertEqual(z12.shape, (2, 256, 768))

class TestGreenBlock(unittest.TestCase):
    def test_forward(self):
        x = torch.randn(1, 768, 256, 256)
        gb = GreenBlock(768, 512)
        y = gb(x)
        self.assertEqual(y.shape, (1, 512, 512, 512))

class TestYellowBlock(unittest.TestCase):
    def test_forward(self):
        x = torch.randn(1, 512, 512, 512)
        yb = YellowBlock(512, 768)
        y = yb(x)
        self.assertEqual(y.shape, (1, 768, 512, 512))

class TestBlueBlock(unittest.TestCase):
    def test_forward(self):
        x = torch.randn(1, 128, 512, 512)
        bb = BlueBlock(128, 64)
        y = bb(x)
        self.assertEqual(y.shape, (1, 64, 1024, 1024))


if __name__ == '__main__':
    unittest.main(argv=[''], verbosity=2, exit=False)
    x = torch.randn(1, 3, 256, 256)
    unetr = UNETR(256, num_classes=5)
    y = unetr(x)
    print(y.shape)

