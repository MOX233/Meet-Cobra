"""Unit tests for chronological stateful TBPTT data and training helpers."""
import unittest
import numpy as np
import torch
from experiment.prepare_stateful_trajectories import frame_values
from experiment.train_stateful_tbptt import noisy_features, forward_valid, run_epoch
from experiment.train_finite_window_vehicle_split import finite_window_samples, batch_arrays
from experiment.vehicle_split import create_vehicle_split
from utils.NN_utils import BeamPredictionLSTMModel, BestGainPredictionLSTMModel
from utils.beam_utils import generate_dft_codebook


class StatefulTBPTTTests(unittest.TestCase):
    def test_vehicle_split_is_disjoint_complete_and_reproducible(self):
        vehicles = np.array(["3", "1", "2", "1", "4", "5"])
        first = create_vehicle_split(vehicles, 0.6, 20)
        second = create_vehicle_split(vehicles, 0.6, 20)
        self.assertTrue(np.array_equal(first[0], second[0]))
        self.assertTrue(np.array_equal(first[1], second[1]))
        self.assertFalse(set(first[0]) & set(first[1]))
        self.assertEqual(set(first[0]) | set(first[1]), set(vehicles))

    def test_finite_windows_remain_inside_selected_trajectories(self):
        data = {
            "offsets": np.array([0, 12, 24]),
            "clean_csi": np.arange(24 * 2).reshape(24, 2).astype(np.complex64),
            "beam": np.arange(24 * 4).reshape(24, 4),
        }
        targets, lengths = finite_window_samples(
            data, np.array([1]), history_length=10, augmentation_ratio=2, seed=20
        )
        self.assertTrue(np.all(targets >= 12))
        self.assertTrue(np.all(targets < 24))
        self.assertEqual(len(targets), 24)
        clean, labels = batch_arrays(data, "beam", targets, lengths, 10)
        self.assertEqual(clean.shape, (24, 10, 2))
        np.testing.assert_array_equal(labels, data["beam"][targets])
        for row, (target, length) in enumerate(zip(targets, lengths)):
            np.testing.assert_array_equal(
                clean[row, :length], data["clean_csi"][target - length + 1:target + 1]
            )

    def test_fft_targets_match_repository_dft(self):
        rng = np.random.default_rng(20)
        records = [{"h": (rng.normal(size=(8,4,32)) + 1j*rng.normal(size=(8,4,32))).astype(np.complex64)} for _ in range(3)]
        clean, beam, desired, interference = frame_values(records)
        tx, rx = generate_dft_codebook(32), generate_dft_codebook(8)
        for i, record in enumerate(records):
            h = record["h"]
            response = np.abs(((h @ tx).T.conj() @ rx).transpose(1,0,2)).reshape(4,-1)
            np.testing.assert_array_equal(beam[i], response.argmax(-1))
            np.testing.assert_allclose(desired[i], 20*np.log10(response.max(-1)/16+1e-9), atol=2e-5)
            np.testing.assert_allclose(interference[i], 20*np.log10(np.abs(h).max((0,2))+1e-9), atol=2e-5)
            expected = (np.sqrt(.1)*(h @ tx)[:,:,::4].sum(-2).reshape(-1)).astype(np.complex64)
            np.testing.assert_allclose(clean[i], expected, rtol=2e-5, atol=2e-7)

    def test_zero_noise_preprocessing(self):
        value = np.ones((2,3,64),dtype=np.complex64)
        actual = noisy_features(value, torch.device('cpu'), torch.Generator().manual_seed(1), noise_power=0)
        self.assertEqual(actual.shape,(2,3,128))
        torch.testing.assert_close(actual[...,:64],torch.full((2,3,64),7.0))
        torch.testing.assert_close(actual[...,64:],torch.zeros(2,3,64))

    def test_all_timestep_heads_match_last_forward(self):
        torch.manual_seed(20)
        for task,model in [('beam',BeamPredictionLSTMModel(128,4,256)),('desired_gain',BestGainPredictionLSTMModel(128,4))]:
            model.eval();x=torch.randn(3,10,128);mask=torch.ones(3,10,dtype=torch.bool)
            with torch.inference_mode():
                out,_=model.lstm_layers(x)
                all_output=forward_valid(model,out,mask,task).reshape(3,10,*model(x).shape[1:])
                torch.testing.assert_close(all_output[:,-1],model(x),rtol=1e-5,atol=1e-5)

    def test_singleton_training_tail_is_safely_omitted(self):
        model = BestGainPredictionLSTMModel(128,4)
        data = {
            'clean_csi': np.ones((11,64),dtype=np.complex64),
            'desired_gain': np.zeros((11,4),dtype=np.float32),
            'offsets': np.array([0,11]),
        }
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4)
        result=run_epoch(model,'desired_gain',data,np.array([0]),torch.device('cpu'),10,1,
                         optimizer,False,20,1.0,True)
        self.assertEqual(result['targets'],40)
        self.assertEqual(result['skipped_singleton_targets'],1)


if __name__=='__main__':
    unittest.main()
