from sys import argv


def parse_dialogs(filename):
    dialogs = []
    with open(filename, 'r') as f:
        dialog = []
        for line in f:
            if line.strip() == '':
                dialogs.append(dialog)
                dialog = []
            else:
                user_utt, bot_utt = line.strip().split('\t')
                utt_num = user_utt.split(' ')[0]
                user_utt = ' '.join(user_utt.split(' ')[1:])
                dialog.append((utt_num, user_utt, bot_utt))
    return dialogs


# TODO: Add different parse modes (context as 1 response or whole previous dialog)
if __name__ == '__main__':
    filename = argv[1]
    dialogs = parse_dialogs(filename)
    for dialog in dialogs:
        for _, user_utt, bot_utt in dialog:
            print('{}\t{}'.format(user_utt, bot_utt))
