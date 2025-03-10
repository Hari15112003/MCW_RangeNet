

import argparse
import subprocess
import datetime
import yaml
from shutil import copyfile
import os
import shutil
import __init__ as booger

from modules.user import *


def inference(dataset,log,model,config):

  # open arch config file
  try:
    print("Opening arch config file from %s" % model)
    ARCH = yaml.safe_load(open(model + "/arch_cfg.yaml", 'r'))
  except Exception as e:
    print(e)
    print("Error opening arch yaml file.")
    quit()

  

  # open data config file
  try:
    print("Opening data config file from %s" % model)
    DATA = yaml.safe_load(open(model + "/data_cfg.yaml", 'r'))
  except Exception as e:
    print(e)
    print("Error opening data yaml file.")
    quit()

  # create log folder
  try:
    if os.path.isdir(log):
      shutil.rmtree(log)
    os.makedirs(log)
    os.makedirs(os.path.join(log, "sequences"))
    for seq in DATA["split"]["train"]:
      seq = '{0:02d}'.format(int(seq))
      # print("train", seq)
      os.makedirs(os.path.join(log, "sequences", seq))
      os.makedirs(os.path.join(log, "sequences", seq, "predictions"))
    for seq in DATA["split"]["valid"]:
      seq = '{0:02d}'.format(int(seq))
      # print("valid", seq)
      os.makedirs(os.path.join(log, "sequences", seq))
      os.makedirs(os.path.join(log, "sequences", seq, "predictions"))
    for seq in DATA["split"]["test"]:
      seq = '{0:02d}'.format(int(seq))
      # print("test", seq)
      os.makedirs(os.path.join(log, "sequences", seq))
      os.makedirs(os.path.join(log, "sequences", seq, "predictions"))
  except Exception as e:
    print(e)
    print("Error creating log directory. Check permissions!")
    raise

  except Exception as e:
    print(e)
    print("Error creating log directory. Check permissions!")
    quit()

  # does model folder exist?
  if os.path.isdir(model):
    print("model folder exists! Using model from %s" % (model))
  else:
    print("model folder doesnt exist! Can't infer...")
    quit()

  # create user and infer dataset
  user = User(ARCH, DATA, dataset, log, model,config)
  if config['auto_quant']:
    print("\n PERFORMING AUTOQUANT")
    user.autoquant()
  if config["quantize"]:
    print("\nPTQ:")
    user.ptq()
  else:
    user.infer()
