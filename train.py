import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F



def train(net,)

net = GoogLeNet().cuda()
optimizer = optim.SGD(net.parameters(),lr=1e-3)
criterion = nn.CrossEntropyLoss()
epochs = 25
total_batch = len(train_dataloader)

train_loss_list = []
val_loss_list = []

for epoch in range(epochs):
  train_loss = 0.0
  for i, data in enumerate(train_dataloader):
      x, label = data
      if is_cuda:
          x = x.cuda()
          label = label.cuda()
        
      optimizer.zero_grad()
      model_output = net(x)

      loss = criterion(model_output, label)
      loss.backward()
      optimizer.step()
      
      train_loss += loss.item() # 이터레이션 수만큼 로스 더함
  
      del loss # del : 메모리 이슈 솔루션
      del model_output
        
      # 학습과정 출력
      if i+1 == total_batch: # 매 에폭의 마지막 이터레이션
          with torch.no_grad(): # 그래디언트 업그레이드 없이. 즉 데이터 넣어서 값만 보고 말겠다는 얘기다
              val_loss = 0.0
              for j, val in enumerate(val_dataloader):
                  val_x, val_label = val
                  if is_cuda:
                      val_x = val_x.cuda()
                      val_label = val_label.cuda()
                  val_output = net(val_x)
                  loss = criterion(val_output, val_label)
                  val_loss += loss.item()
                       
          print("epoch: {}/{} | step: {}/{} | train loss: {:.4f} | val loss: {:.4f}".format(
              epoch+1, epochs, i+1, total_batch, train_loss/total_batch, val_loss/len(val_dataloader)
          ))            
            
          train_loss_list.append(train_loss/total_batch) # 각 트레인 로스는 이터레이션 회수만큼 더해졌을테니 평균을 내준다
          val_loss_list.append(val_loss/len(val_dataloader)) # 너 역시
          train_loss = 0.0