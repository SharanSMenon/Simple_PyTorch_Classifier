from sclassifier import train_classifier_on_directory

if __name__ == "__main__":
    train_classifier_on_directory("../data/antsvbees", test=False, data_in_train_dir=True)