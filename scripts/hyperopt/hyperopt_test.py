from hyperopt import tpe, hp, Domain, Trials, STATUS_OK, JOB_STATE_DONE
import math
import numpy as np

def suggest_model(sid, trials):
  sug = tpe.suggest((sid,), domain, trials, rstate.randint(2 ** 31 - 1))[0]
  suggestions[sid] = sug
  return sug

def add_result(sid, err, trials):
  #Results should have the format "id\tvalue"
  sid = int(sid)
  err = float(err)
  results[sid] = suggestions[sid]
  results[sid]["result"] = {"loss": err, "status": STATUS_OK}
  trials.insert_trial_doc(results[sid])
  trials.refresh()

#Set up a parameter space
space = (hp.uniform('lr', -10, 10))

domain = Domain(lambda i: i, space)
rstate = np.random.RandomState(42)
suggestions = {}
results = {}
trials = Trials()
sid = 0
for i in range(10000):
  # print i
  suggest_model(i, trials)
  x = suggestions[i]['misc']['vals']['lr']
  print x[0]
  err = x[0] * x[0]
  # print err
  add_result(i, err, trials)
