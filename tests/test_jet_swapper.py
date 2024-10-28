import unittest
import pandas as pd
import numpy as np
import torch
from hepytorch.preprocessors.jet_swapper import JetSwapper


class TestJetSwapper(unittest.TestCase):
    def setUp(self):
        # Create a sample DataFrame for testing
        data = {
            "j_1px": np.arange(10),
            "j_1py": np.arange(10),
            "j_1pz": np.arange(10),
            "j_1mass": np.arange(10),
            "l_1px": np.arange(10),
            "l_1py": np.arange(10),
            "l_1pz": np.arange(10),
            "l_1mass": np.arange(10),
            "j_2px": np.arange(10, 20),
            "j_2py": np.arange(10, 20),
            "j_2pz": np.arange(10, 20),
            "j_2mass": np.arange(10, 20),
            "l_2px": np.arange(10),
            "l_2py": np.arange(10),
            "l_2pz": np.arange(10),
            "l_2mass": np.arange(10),
            "mex": np.arange(10),
            "mey": np.arange(10),
        }
        self.df = pd.DataFrame(data)
        self.swapper = JetSwapper()

    def test_data_shape(self):
        result = self.swapper.data(self.df)
        self.assertEqual(result.shape, (10, 18))

    def test_data_type(self):
        result = self.swapper.data(self.df)
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.dtype, torch.float)

    def test_swap_logic(self):
        self.swapper.data(self.df)
        mask = self.swapper.mask
        swapped_df = self.swapper.data

        for index, row in swapped_df.iterrows():
            if self.swapper.mask[index]:
                self.assertEqual(row["j_1px"], self.df.loc[index, "j_2px"])
                self.assertEqual(row["j_1py"], self.df.loc[index, "j_2py"])
                self.assertEqual(row["j_1pz"], self.df.loc[index, "j_2pz"])
                self.assertEqual(row["j_1mass"], self.df.loc[index, "j_2mass"])
                self.assertEqual(row["j_2px"], self.df.loc[index, "j_1px"])
                self.assertEqual(row["j_2py"], self.df.loc[index, "j_1py"])
                self.assertEqual(row["j_2pz"], self.df.loc[index, "j_1pz"])
                self.assertEqual(row["j_2mass"], self.df.loc[index, "j_1mass"])
            else:
                self.assertEqual(row["j_1px"], self.df.loc[index, "j_1px"])
                self.assertEqual(row["j_1py"], self.df.loc[index, "j_1py"])
                self.assertEqual(row["j_1pz"], self.df.loc[index, "j_1pz"])
                self.assertEqual(row["j_1mass"], self.df.loc[index, "j_1mass"])
                self.assertEqual(row["j_2px"], self.df.loc[index, "j_2px"])
                self.assertEqual(row["j_2py"], self.df.loc[index, "j_2py"])
                self.assertEqual(row["j_2pz"], self.df.loc[index, "j_2pz"])
                self.assertEqual(row["j_2mass"], self.df.loc[index, "j_2mass"])


if __name__ == "__main__":
    unittest.main()
