import pickle
with open("./resnet/resnet110_no_shortcut_epoch.txt", "rb") as fp:   # Unpickling 
  x= pickle.load(fp)
with open("./resnet/resnet110_no_shortcut_loss.txt", "rb") as fp:   # Unpickling
  y= pickle.load(fp)
with open("./resnet/resnet110_no_shortcut_epoch1.txt", "rb") as fp:   # Unpickling 
  u= pickle.load(fp)
with open("./resnet/resnet110_no_shortcut_acc.txt", "rb") as fp:   # Unpickling
  v= pickle.load(fp)
print(x,y)
print(u,v)
