from feature_extraction import create_grads_dataset, create_losses_dataset
from classification import generate_supervised_results, generate_unsupervised_results

def run_experiments(zoos, msb=False, malware_name="malware_12584bytes", n_jobs=-1):
    for zoo in zoos:
        create_grads_dataset(zoo, malware_name=malware_name, msb=msb, n_jobs=n_jobs)
        create_losses_dataset(zoo, malware_name=malware_name, msb=msb)

    generate_supervised_results(zoos=zoos, msb=msb, n_jobs=n_jobs)
    generate_unsupervised_results(zoos=zoos, msb=msb)


"""You need to choose which zoos to run experiments on:
    Options are:
        cifar10
        mnist
        stl10
        svhn
"""

if __name__ == "__main__":
    n_jobs = -1 # Beware: -1 means use all available cores
    malware_name = "malware_12584bytes" # Enter malware file name here
    zoos = ["mnist", "stl10", "cifar10", "svhn"]
    msb = True

    run_experiments(zoos, msb=msb, malware_name=malware_name, n_jobs=n_jobs)


