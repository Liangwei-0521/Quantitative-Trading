import json5
import argparse


def args_parameters():
    with open(r'E:\Code_Project\Agent with imitation\config.json', encoding='UTF-8') as f:
        Config = json5.load(f)
        json5.dumps(Config, indent=4, ensure_ascii=False)

    parser = argparse.ArgumentParser()
    parser.add_argument('--cost', type=float, default=Config['cost'])
    parser.add_argument('--window_length', type=int, default=Config['window_length'])
    parser.add_argument('--timeperiod', type=int, default=Config['timeperiod'])
    parser.add_argument('--balance', type=int, default=Config['balance'])
    parser.add_argument('--device', type=str, default=Config['device'])
    parser.add_argument('--state_dim', type=int, default=Config['state_dim'])
    parser.add_argument('--hidden_dim', type=int, default=Config['hidden_dim'])
    parser.add_argument('--action_dim', type=int, default=Config['action_dim'])
    parser.add_argument('--learning_rate', type=int, default=Config['learning_rate'])
    parser.add_argument('--times', type=int, default=Config['times'], help='the behavior clone agent')
    parser.add_argument('--batch_size', type=int, default=Config['batch_size'])
    parser.add_argument('--start', type=int, default=Config['start'], help='Time starting point')
    parser.add_argument('--gamma', type=float, default=Config['gamma'])
    parser.add_argument('--epsilon', type=float, default=Config['epsilon'])
    parser.add_argument('--target_update', type=int, default=Config['target_update'])
    parser.add_argument('--lambda_1', type=float, default=Config['lambda_1'])
    parser.add_argument('--lambda_2', type=float, default=Config['lambda_2'])

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    parser = args_parameters()
    print(type(parser.window_length), parser.window_length)
