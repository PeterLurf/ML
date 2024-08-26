import unittest

def leaky_relu(z: float, alpha: float = 0.01) -> float:
    return z if z >= 0 else alpha * z

class TestLeakyReLU(unittest.TestCase):
    def test_positive(self):
        self.assertEqual(leaky_relu(5), 5)
    
    def test_negative(self):
        self.assertEqual(leaky_relu(-5), -0.05)
    
    def test_zero(self):
        self.assertEqual(leaky_relu(0), 0)

if __name__ == '__main__':
    unittest.main()