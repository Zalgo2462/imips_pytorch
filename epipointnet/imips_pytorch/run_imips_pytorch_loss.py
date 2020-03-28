import os.path

import numpy as np
import torch

import epipointnet.imips_pytorch.trainer

"""
run_imips_pytorch_loss.py runs trainer.py:ImipsTrainer.__loss (the re-implementation of the IMIPS loss routine) against
1000 test cases of synthetic IMIPS network output on 256 image patches each. It reads the test case numpy files
(scores.npy and labels.npy) from ./test_cases/test_case_####. If the test case data does not exist, the script
will create it. The imips loss and the sub-losses for each case are then saved to
./test_cases/test_case_####/losses-pytorch.npy.  

This script is meant to be run along side the similar script run_imis_tf_loss.py.
"""

for i in range(1000):
    test_case_dir = "./test_cases/test_case_" + str(i) + "/"

    if os.path.exists(test_case_dir):
        print("Loading data")
        scores = np.load(test_case_dir + "scores.npy")
        labels = np.load(test_case_dir + "labels.npy")
    else:
        print("Creating data")
        scores = np.random.rand(256, 1, 1, 128)
        labels = np.random.rand(256) > .5
        try:
            os.mkdir(test_case_dir)
        except:
            pass
        np.save(test_case_dir + "scores.npy", scores)
        np.save(test_case_dir + "labels.npy", labels)

    maximizer_outputs = torch.tensor(scores[0:scores.shape[0] // 2], dtype=torch.float32).cuda().permute((0, 3, 1, 2))
    correspondence_outputs = torch.tensor(scores[scores.shape[0] // 2:scores.shape[0]],
                                          dtype=torch.float32).cuda().permute((0, 3, 1, 2))
    inlier_labels = torch.tensor(labels[0:labels.shape[0] // 2]).cuda().to(torch.uint8)
    outlier_labels = torch.tensor(labels[labels.shape[0] // 2:labels.shape[0]]).cuda().to(torch.uint8)
    eps = torch.tensor(1e-4).cuda()

    loss, outlier_correspondence_loss, inlier_loss, \
    outlier_maximizer_loss, unaligned_maximizer_loss = epipointnet.imips_pytorch.trainer.ImipTrainer.loss(
        maximizer_outputs, correspondence_outputs, inlier_labels, outlier_labels, eps
    )

    loss = loss.cpu().numpy()
    outlier_correspondence_loss = outlier_correspondence_loss.cpu().numpy()
    inlier_loss = inlier_loss.cpu().numpy()
    outlier_maximizer_loss = outlier_maximizer_loss.cpu().numpy()
    unaligned_maximizer_loss = unaligned_maximizer_loss.cpu().numpy()

    losses = np.array(
        [loss, outlier_correspondence_loss, inlier_loss, outlier_maximizer_loss, unaligned_maximizer_loss]
    )

    np.save(test_case_dir + "losses-pytorch.npy", losses)
