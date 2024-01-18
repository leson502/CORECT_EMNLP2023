from comet_ml import Experiment, Optimizer

import argparse
import torch
import os
import corect

log = corect.utils.get_logger()

def main(args):
    corect.utils.set_seed(args.seed)

    if args.emotion:
        args.data = os.path.join(
            args.data_root,
            args.data_dir_path,
            args.dataset,
            "data_" + args.dataset + "_" + args.emotion + ".pkl",
        )
    else:
        args.data = os.path.join(
            args.data_root, args.data_dir_path, args.dataset, "data_" + args.dataset + ".pkl"
        )

    # load data
    log.debug("Loading data from '%s'." % args.data)

    data = corect.utils.load_pkl(args.data)
    log.info("Loaded data.")

    trainset = corect.Dataset(data["train"], args)
    devset = corect.Dataset(data["dev"], args)
    testset = corect.Dataset(data["test"], args)

    log.debug("Building model...")
    
    model_file = args.data_root + "/model_checkpoints/model.pt"
    model = corect.CORECT(args).to(args.device)
    opt = corect.Optim(args.learning_rate, args.max_grad_value, args.weight_decay)
    opt.set_parameters(model.parameters(), args.optimizer)
    sched = opt.get_scheduler(args.scheduler)

    coach = corect.Coach(trainset, devset, testset, model, opt, sched, args)
    if not args.from_begin:
        ckpt = torch.load(model_file)
        coach.load_ckpt(ckpt)
        print("Training from checkpoint...")

    # Train
    log.info("Start training...")
    ret = coach.train()

    # Save.
    checkpoint = {
        "best_dev_f1": ret[0],
        "best_epoch": ret[1],
        "best_state": ret[2],
    }
    torch.save(checkpoint, model_file)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="train.py")
    parser.add_argument(
        "--dataset",
        type=str,
        # required=True,
        default="iemocap_4",
        choices=["iemocap", "iemocap_4","mosei"],
        help="Dataset name.",
    )

    parser.add_argument(
        "--data_dir_path", type=str, help="Dataset directory path", default="./data"
    )

    # Training parameters
    parser.add_argument(
        "--from_begin", action="store_true", help="Training from begin.", default=False
    )

    parser.add_argument("--device", type=str, default="cuda", help="Computing device.")
    parser.add_argument(
        "--epochs", default=1, type=int, help="Number of training epochs."
    )
    parser.add_argument("--batch_size", default=10, type=int, help="Batch size.")
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adam",
        choices=["sgd", "rmsprop", "adam", "adamw"],
        help="Name of optimizer.",
    )
    parser.add_argument(
        "--scheduler", type=str, default="reduceLR", help="Name of scheduler."
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.00025, help="Learning rate."
    )
    parser.add_argument(
        "--weight_decay", type=float, default=1e-8, help="Weight decay."
    )
    parser.add_argument(
        "--max_grad_value",
        default=-1,
        type=float,
        help="""If the norm of the gradient vector exceeds this,
                        normalize it to have the norm equal to max_grad_norm""",
    )
    parser.add_argument("--drop_rate", type=float, default=0.5, help="Dropout rate.")

    # Model parameters
    parser.add_argument(
        "--wp",
        type=int,
        default=11,
        help="Past context window size. Set wp to -1 to use all the past context.",
    )
    parser.add_argument(
        "--wf",
        type=int,
        default=9,
        help="Future context window size. Set wp to -1 to use all the future context.",
    )
    
    parser.add_argument(
        "--hidden_size", type=int, default=100, help="Hidden size."
    )
    parser.add_argument(
        "--rnn",
        type=str,
        default="transformer",
        choices=["lstm", "transformer", "ffn"],
        help="Type of RNN encoder cell.",
    )

    parser.add_argument(
        "--class_weight",
        action="store_true",
        default=False,
        help="Use class weights in nll loss.",
    )

    # Modalities
    """ Modalities effects:
        -> dimentions of input vectors in dataset.py
        -> number of heads in transformer_conv in UnimodalEncoder.py"""
    parser.add_argument(
        "--modalities",
        type=str,
        default="atv",
        # required=True,
        choices=["a", "t", "v", "at", "tv", "av", "atv"],
        help="Modalities",
    )

    parser.add_argument(
        "--gcn_conv",
        type=str,
        default="rgcn",
        # required=True,
        choices=["rgcn"],
        help="Graph convolution layer",
    )

    # emotion
    parser.add_argument(
        "--emotion", type=str, default=None, help="emotion class for mosei"
    )

    
    parser.add_argument("--encoder_nlayers", type=int, default=2)
    parser.add_argument("--graph_transformer_nheads", type=int, default=7)
    parser.add_argument("--use_highway", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=24, help="Random seed.")
    parser.add_argument(
        "--log_in_comet",
        action="store_true",
        default=False,
        help="Logs the experiment data to comet.ml",
    )
    parser.add_argument(
        "--comet_api_key",
        type=str,
        help="comet api key, required for logging experiments on comet.ml",
    )
    parser.add_argument(
        "--comet_workspace",
        type=str,
        help="comet comet_workspace, required for logging experiments on comet.ml",
    )

    parser.add_argument(
        "--data_root",
        type=str,
        default=".",
        help="data root of folder",
    )

    parser.add_argument(
        "--edge_type",
        default="temp_multi",
        choices=("temp_multi", "multi", "temp"),
        help="Choices edge contruct type",
    )

    parser.add_argument(
        "--use_speaker",
        action="store_true",
        default=False,
        help="Use speakers attribute",
    )

    parser.add_argument(
        "--no_gnn",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--use_graph_transformer",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--use_crossmodal",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--crossmodal_nheads",
        type=int,
        default=2,
        help="number attention heads in crossmodal attention block"
    )

    parser.add_argument(
        "--num_crossmodal",
        type=int,
        default=2,
        help="number crossmodal block"
    )

    parser.add_argument(
        "--self_att_nheads",
        type=int,
        default=2,
        help="number attention heads in self attention block",
    )

    parser.add_argument(
        "--num_self_att",
        type=int,
        default=3,
        help="number self attention block"
    )

    parser.add_argument("--tag", type=str, default="normalexperiment")

    args = parser.parse_args()

    args.dataset_embedding_dims = {
        "iemocap": {
            "a": 100,
            "t": 768,
            "v": 512,
        },
        "iemocap_4": {
            "a": 100,
            "t": 768,
            "v": 512,
        },
        "mosei": {
            "a": 80,
            "t": 768,
            "v": 35,
        },
    }

    log.debug(args)

    if args.log_in_comet:
        experiment = corect.Logger(
                api_key=args.comet_api_key,
                project_name="rtgraph",
                workspace=args.comet_workspace,
                auto_param_logging=False,
                auto_metric_logging=False,
                )

        experiment.add_tag(args.tag)
        experiment.log_parameters(args)
        print(experiment)

    main(args)
