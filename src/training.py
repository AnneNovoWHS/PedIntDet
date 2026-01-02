from tqdm import tqdm          

from src.evaluating import evaluater
from src.logging import Logger

##################################################################################################

def trainer(trainloader, valloader, model, device, cfg):
    
    # setup tensorboard logging
    log = Logger(cfg)

    #=================================================================================
    
    # train loop
    for epoch in range(model.epoch+1, model.cfg.training.epochs):
        print(f'EPOCH {epoch+1}:')

        # get model ready
        model.on_epoch_start()

        # go through all data
        for i, (inputs, labels) in  enumerate(tqdm(trainloader, desc=f'training')):
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = {k: v.to(device) for k, v in labels.items()}

            # forward
            outputs = model(inputs)           

            # update step 
            training_logs = model.train_step(outputs, labels)

            # log norm during training
            log.write_train_step_tensorboard(dict=training_logs, step=epoch*len(trainloader) + i, split='train', phase='training')

        #=================================================================================

        # evaluate model
        train_metrics = evaluater(model, trainloader, device, 'train')
        val_metrics = evaluater(model, valloader, device, 'val')

        #=================================================================================

        # write to tensorboard
        log.write_eval_tensorboard(dict=train_metrics, step=int((epoch+1)*len(trainloader)), split='train', phase='evaluation')
        log.write_eval_tensorboard(dict=val_metrics, step=int((epoch+1)*len(trainloader)), split='val', phase='evaluation')

        # save checkpoint each epoch
        log.save_checkpoint(model)
        
        #=================================================================================

        # print prediction metrics for training and validation data after warmup
        for key in train_metrics.keys():
            print(key.upper())
            for metric, train_val in train_metrics[key].items():
                val_val = val_metrics[key].get(metric, float('nan'))
                print(f"{metric.title():<10}  Train: {train_val:.3f}  Val: {val_val:.3f}")
