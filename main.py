
import torch
import dataset
from model import LeNet5, regularized_LeNet5, CustomMLP
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import matplotlib.pyplot as plt
import time



#-------------------------train function------------------------------------------------------
def train(model, trn_loader, device, criterion, optimizer):
    """ Train function

    Args:
        model: network
        trn_loader: torch.utils.data.DataLoader instance for training
        device: device for computing, cpu or gpu
        criterion: cost function
        optimizer: optimization method, refer to torch.optim

    Returns:
        trn_loss: average loss value
        acc: accuracy
    """
    
    
    total_batch = len(trn_loader)
    trn_loss = 0; acc = 0
        
    for batch_idx, batch in enumerate(trn_loader) :
        imgs, labels = batch
        imgs = imgs.to(device) 
        labels = labels.to(device)
        optimizer.zero_grad()
        output = model(imgs)
        cost = criterion(output, labels)
        cost.backward()
        optimizer.step()
        
        # caculate accuracy
        accuracy = (torch.argmax(output, dim=1)==labels).sum()/len(imgs)
        acc += accuracy

        # get loss
        trn_loss += cost.item()
        
    acc = round(acc.item()/total_batch, 3); trn_loss = round(trn_loss/total_batch, 3)

    return trn_loss, acc

#-------------------------validation function-----------------------------------------------------
def test(model, tst_loader, device, criterion):
    """
    Test function

    Args:
        model: network
        tst_loader: torch.utils.data.DataLoader instance for testing
        device: device for computing, cpu or gpu
        criterion: cost function

    Returns:
        tst_loss: average loss value
        acc: accuracy
    """
    total_batch = len(tst_loader)
    
    acc = 0; tst_loss = 0
    for batch_idx, batch in enumerate(tst_loader) :
        with torch.no_grad() : 
            imgs, labels = batch
            imgs, labels = imgs.to(device), labels.to(device)
            output = model(imgs)
            cost = criterion(output, labels)
            
            # get test accuracy
            accuracy = (torch.argmax(output, dim=1) == labels).sum() / len(imgs)
            acc += accuracy
            
            # get loss
            tst_loss += cost.item()
    
    tst_loss = round(tst_loss/total_batch, 3)
    acc = round(acc.item()/total_batch, 3)


    return tst_loss, acc


#------------------------------main function-------------------------------------------------------
def main():
    """ Main function

        Here, you should instantiate
        1) Dataset objects for training and test datasets
        2) DataLoaders for training and testing
        3) model
        4) optimizer: SGD with initial learning rate 0.01 and momentum 0.9
        5) cost function: use torch.nn.CrossEntropyLoss

    """

    # ========== 1. data load ==========
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.1307], [0.3081])])
    mnist_dataset = dataset.MNIST(data_dir = 'data/train.tar', transform=transform)
    train_dataset, test_dataset = random_split(mnist_dataset, [50000, 10000])

    train_data = DataLoader(train_dataset, batch_size=64)
    test_data = DataLoader(test_dataset, batch_size=64)

    
    # ========== 2. Lenet 5 model ==========
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    training_epochs = 10
    
    lenet_model = LeNet5().to(device)
    lenet_optimizer = torch.optim.SGD(lenet_model.parameters(), lr=0.01, momentum=0.9)
    lenet_cost_function = torch.nn.CrossEntropyLoss().to(device)
    
    
    print('Lenet 5 training start ')
    lenet_time = time.time()
    lenet_trn_loss, lenet_trn_acc, lenet_tst_loss, lenet_tst_acc = [],[],[],[]
    for epoch in range(training_epochs) :
        lenet_train_loss, lenet_train_acc = train(model=lenet_model, trn_loader=train_data, device=device, criterion=lenet_cost_function, 
                                                  optimizer=lenet_optimizer)
        lenet_test_loss, lenet_test_acc = test(model=lenet_model, tst_loader=test_data, device=device, criterion=lenet_cost_function)
        
        lenet_trn_loss.append(lenet_train_loss); lenet_trn_acc.append(lenet_train_acc)
        lenet_tst_loss.append(lenet_test_loss); lenet_tst_acc.append(lenet_test_acc)
        
        print('epochs {} training loss {}  training accuracy {} validation loss {}  validation accuracy {}'.format(epoch, lenet_train_loss, lenet_train_acc, 
                                                                                                                   lenet_test_loss, lenet_test_acc))
        if epoch+1 == 10 :
            print('lenet execution time : {}'.format(time.time() - lenet_time))
        
    # ========== 3. Regularized Lenet 5 model ==========
    regularized_lenet_model = regularized_LeNet5().to(device)
    regularized_lenet_optimizer = torch.optim.SGD(regularized_lenet_model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.001)
    regularized_lenet_cost_function = torch.nn.CrossEntropyLoss().to(device)
    
    print('Regularized Lenet 5 training start ')
    r_lenet_time = time.time()
    r_lenet_trn_loss, r_lenet_trn_acc, r_lenet_tst_loss, r_lenet_tst_acc = [],[],[],[]
    for epoch in range(training_epochs) :
        regularized_lenet_train_loss, regularized_lenet_train_acc = train(model=regularized_lenet_model, trn_loader=train_data, device=device, 
                                                                          criterion=regularized_lenet_cost_function, optimizer=regularized_lenet_optimizer)
        regularized_lenet_test_loss, regularized_lenet_test_acc = test(model=regularized_lenet_model, tst_loader=test_data, device=device, 
                                                                       criterion=regularized_lenet_cost_function)
        
        r_lenet_trn_loss.append(regularized_lenet_train_loss); r_lenet_trn_acc.append(regularized_lenet_train_acc)
        r_lenet_tst_loss.append(regularized_lenet_test_loss); r_lenet_tst_acc.append(regularized_lenet_test_acc)
        
        print('epochs {} training loss {}  training accuracy {} validation loss {}  validation accuracy {}'.format(epoch, regularized_lenet_train_loss, regularized_lenet_train_acc, 
                                                                                                                   regularized_lenet_test_loss, regularized_lenet_test_acc))
        
        if epoch+1 == 10 :
            print('regularized execution time : {}'.format(time.time() - r_lenet_time))
    
    
    # ========== 4. Custom model Load ==========
    custom_model = CustomMLP().to(device)
    custom_optimizer = torch.optim.SGD(custom_model.parameters(), lr=0.01, momentum=0.9)
    custom_cost_function = torch.nn.CrossEntropyLoss().to(device)
    
    print('Custom model training start')
    custom_time = time.time()
    custom_trn_loss, custom_trn_acc, custom_tst_loss, custom_tst_acc = [],[],[],[]
    for epoch in range(training_epochs) :
        custom_train_loss, custom_train_acc = train(model=custom_model, trn_loader=train_data, device=device, criterion=custom_cost_function, optimizer=custom_optimizer)
        custom_test_loss, custom_test_acc = test(model=custom_model, tst_loader=test_data, device=device, criterion=custom_cost_function)
    
        custom_trn_loss.append(custom_train_loss); custom_trn_acc.append(custom_train_acc)
        custom_tst_loss.append(custom_test_loss); custom_tst_acc.append(custom_test_acc)
        
        print('epochs {} training loss {}  training accuracy {} validation loss {}  validation accuracy {}'.format(epoch, custom_train_loss, custom_train_acc, 
                                                                                                                   custom_test_loss, custom_test_acc))
        
        if epoch+1 == 10 :
            print('custom model execution time : {}'.format(time.time() - custom_time))
    
    
    
    # ========== 5. visualization ==========
    #  make loss and acc list for visualization
    trn_loss = [lenet_trn_loss, r_lenet_trn_loss, custom_trn_loss]
    trn_acc = [lenet_trn_acc, r_lenet_trn_acc, custom_trn_acc]
    tst_loss = [lenet_tst_loss, r_lenet_tst_loss, custom_tst_loss]
    tst_acc = [lenet_tst_acc, r_lenet_tst_acc, custom_tst_acc]
    
    # draw plot
    draw_plot(trn_loss, trn_acc, tst_loss, tst_acc)




#------------------------------draw cost function-------------------------------------------------------
# input : list of each loss and accuracy
def draw_plot(trn_loss, trn_acc, val_loss, val_acc) :
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16,10))
    
    # draw train loss graph 
    axes[0, 0].plot(trn_loss[0], label='lanet'); axes[0, 0].plot(trn_loss[1], label='regularized lanet'); axes[0, 0].plot(trn_loss[2], label='custom')
    axes[0, 0].set_title('train loss function')
    axes[0, 0].legend()
    
    # draw train acc graph
    axes[0, 1].plot(trn_acc[0], label='lanet'); axes[0, 1].plot(trn_acc[1], label='regularized lanet'); axes[0, 1].plot(trn_acc[2], label='custom')
    axes[0, 1].set_title('train accuracy function')
    axes[0, 1].legend()
    
    # draw validation loss graph
    axes[1, 0].plot(val_loss[0], label='lanet'); axes[1, 0].plot(val_loss[1], label='regularized lanet'); axes[1, 0].plot(val_loss[2], label='custom')
    axes[1, 0].set_title('validation loss function')
    axes[1, 0].legend()    
 
    # draw validation acc graph
    axes[1, 1].plot(val_acc[0], label='lanet'); axes[1, 1].plot(val_acc[1], label='regularized lanet'); axes[1, 1].plot(val_acc[2], label='custom')
    axes[1, 1].set_title('validation accuracy function')
    axes[1, 1].legend()
    
    plt.savefig('output_1.png')
    
    
    
if __name__ == '__main__':
    main()
