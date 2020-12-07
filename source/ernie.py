import argparse
import datetime

def update_model_log(model, args, stats, train=True):
    logfile = "./saved_models/" + args.model_dir + "/logfile.txt"
    try:
        with open(logfile, 'r') as original: log = original.readlines()
    except FileNotFoundError:
        log = ["\nModel architecture: \n\n", str(model)]

    with open(logfile, 'w') as modified: 
        towrite = ('*' * 80) + '\n\n'
        if train:
            # We are logging what occurred during training...
            towrite += "TRAINING LOG:\n"
            towrite += "Date: " + datetime.datetime.now().strftime("%Y-%m-%d, %H:%M:%S") + '\n\n'
            towrite += "Settings:\n"
            towrite += "clef_year (19=speeches, 20=covid twitter, or -1=ClaimBuster): " + str(args.clef_year) + "\n"
            towrite += "Total epochs: " + str(args.epochs) + "\n"
            towrite += "Batch size: " + str(args.batch_size) + "\n"
            towrite += "Weights?: " + str(args.weights) + "\n"
            towrite += "Sampler?: " + str(args.sampler) + "\n"
            towrite += "Base layers unfrozen?: " + str(args.unfreeze) + "\n"
            towrite += "Model under the hood?: " + str(args.load) + "\n"
        else:
            towrite += "TEST LOG:\n"
            towrite += "Date: " + datetime.datetime.now().strftime("%Y-%m-%d, %H:%M:%S") + '\n\n'
            towrite += "Settings:\n"
            towrite += "clef_year (19=speeches, 20=covid twitter, or -1=ClaimBuster): " + str(args.clef_year) + "\n"
            if args.train:
                towrite += "Model test after: " + str(args.epochs) + " epochs\n"
            towrite += "Location of model predictions: ./saved_models/" + str(args.model_dir) + "/preds/" + "\n"

        towrite += "\n"
        towrite += "Statistics:\n"
        for key in stats.keys():
            if "Tensor" in str(type(stats[key])):
                towrite += key + str(stats[key].item()) + "\n"
            else:
                towrite += key + str(stats[key]) + "\n"
        
        towrite += "\n"
        towrite += ('*' * 80) + '\n'
        modified.write(towrite)
        modified.writelines(log)
    print("Updated ", logfile)

def printProgressBar(iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
                            
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

def get_args():
    parser = argparse.ArgumentParser(description='Fine tunes RoBERTa on CLEF 2019 or 2020 claim detection datasets... Makes a claim detection model.')
    parser.add_argument('--load', action='store', default='', help='Path of a pyTorch checkpoint file that you would like to load into the model. If none, then RoBERTa base will be loaded ready to fine tune and/or test.')
    parser.add_argument('--train', action='store_true', help='Provide this flag if you would like to train your model.')
    parser.add_argument('--weights', action='store_true', help='Provide this flag if you would like the classes to be weighted when training based on their prevalence in the training data. It is usually recommended that this flagged be passed only when the "--unfreeze" flag is also passed.')
    parser.add_argument('--sampler', action='store_true', help='Provide this flag if you would like a sampler to sample the under-represented class as much as the over-represented class when training.')
    parser.add_argument('--unfreeze', action='store_true', help='Provide this flag if you would like the pre-trained encoder (base) layers of the model trainable. By default, those are frozen, so only the weights of the head layers are optimized.')
    parser.add_argument('--epochs', type=int, action='store', default=3, help="The number of epochs to train your model for. By default, epochs = 3.")
    parser.add_argument('--batch_size', type=int, action='store', default=32, help="The batch size to be used for training and testing. By default, batch size = 32.")
    parser.add_argument('--clef_year', type=int, action='store', default=20, help="The year for the clef CheckThat! check-worthiness challenge to be used for training & testing. Must be one of -1, 19, or 20. By default, clef_year = 20. -1 is for the ClaimBuster dataset.")
    parser.add_argument('--model_dir', action='store', default='latest_model', help="The name of the directory in which to save the trained model / model predictions. By default = latest_model. Do NOT add the trailing '/'.")
    parser.add_argument('--test', action='store_true', help='Provide this flag if you would like to test your model.')
    return parser.parse_args()
