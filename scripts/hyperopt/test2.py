from hyperopt import tpe, hp, Domain, Trials, STATUS_OK, JOB_STATE_DONE
import math
import numpy as np

def suggest_model(sid, trials):
  sug = tpe.suggest((sid,), domain, trials, rstate.randint(2 ** 31 - 1))[0]
  suggestions[sid] = sug
  return sug

#Set up a parameter space
space = hp.choice('model',
  [
    ('logistic',[hp.loguniform('lr0', math.log(1e-3), math.log(1e1)), hp.loguniform('reg0', math.log(1e-4), math.log(1e2))]),
    ('svm',[hp.loguniform('lr1', math.log(1e-3), math.log(1e1)), hp.loguniform('reg1', math.log(1e-4), math.log(1e2))])
  ])

domain = Domain(lambda i: i, space)
rstate = np.random.RandomState(42)
suggestions = {}
results = {}
trials = Trials()
sid = 0
for i in range(10):
  print suggest_model(sid, trials)
