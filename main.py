from src.initializer import get_args, set_seed, set_logger, log_args
from src.dataloader import load_data
from src.model import BSPM
from src.runner import train, test

def main():
    args = get_args()
    set_seed(args.seed)
    set_logger(args.comment)
    log_args(args)
    
    train_data, test_dataloader = load_data(args.dataset, args.batch_size, args.device)
    model = BSPM(args.sharp_solv, args.sharp_step, args.sharp_time,
                 args.ideal_cutoff, args.ideal_weight,
                 args.early_merge, args.sharp_off, args.point_combi)
    train(model, train_data, args.device)
    test(model, test_dataloader, args.metrics, args.top_ks)

if __name__ == '__main__':
    main()