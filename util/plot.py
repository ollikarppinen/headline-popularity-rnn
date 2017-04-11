import pandas as pd
import matplotlib.pyplot as plt
import os

def dense_layer(data,path):
    plt.scatter(data[0],data[1],c=data[3],s=1, cmap='cool')
    plt.savefig(os.path.join(path, 'scatter.pdf'))
    plt.clf()
    plt.hist2d(data[0],data[1],bins=200)
    plt.savefig(os.path.join(path, 'hist.pdf'))
    plt.clf()
    data.to_csv(os.path.join(path, 'dense.csv'))

def epochs(epoch_history, path):
    df = pd.DataFrame()
    # print(epoch_history)
    df['acc'] = epoch_history.history['acc']
    df['loss'] = epoch_history.history['loss']
    df['val_acc'] = epoch_history.history['val_acc']
    df['val_loss'] = epoch_history.history['val_loss']
    df.to_csv(os.path.join(path, 'epoch_history.csv'))
    # accuracy history
    plt.plot(df['acc'])
    plt.plot(df['val_acc'])
    plt.title('mallin tarkkuus')
    plt.ylabel('tarkkuus')
    plt.xlabel('epookki')
    plt.legend(['koulutus', 'validointi'], loc='upper left')
    plt.savefig(os.path.join(path, 'epoch_accuracy.pdf'))
    plt.clf()
    # loss history
    plt.plot(df['loss'])
    plt.plot(df['val_loss'])
    plt.title('mallin virhe')
    plt.ylabel('virhe')
    plt.xlabel('epookki')
    plt.legend(['koulutus', 'validointi'], loc='upper left')
    plt.savefig(os.path.join(path, 'epoch_loss.pdf'))
    plt.clf()

def batches(batch_history, path):
    df = pd.DataFrame()
    df['loss'] = batch_history.losses
    df['acc'] = batch_history.accs
    df.to_csv(os.path.join(path, 'epoch_history.csv'))
    # batch testing accuracy history
    plt.plot(batch_history.accs)
    plt.title('mallin tarkkuus')
    plt.ylabel('tarkkuus')
    plt.xlabel('erä')
    plt.legend(['koulutus'], loc='upper left')
    plt.savefig(os.path.join(path, 'batch_accuracy.pdf'))
    plt.clf()
    # batch testing loss history
    plt.plot(batch_history.losses)
    plt.title('mallin virhe')
    plt.ylabel('virhe')
    plt.xlabel('erä')
    plt.legend(['koulutus'], loc='upper left')
    plt.savefig(os.path.join(path, 'batch_loss.pdf'))
