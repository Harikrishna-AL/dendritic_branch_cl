import numpy as np
from bdh_cl.scripts.mnist_short_term_context_test import run_stream,stream_order


def fixture():
 rng=np.random.default_rng(3);n=40
 z=rng.normal(size=(n,168,4)).astype('float32');cur=rng.uniform(size=z.shape).astype('float32');cur/=cur.sum(-1,keepdims=True)
 cache=dict(z=z,current=cur,pseudo=cur[:,::-1].copy(),supervised=cur[:,:,::-1].copy());y=np.arange(n)%10;w=rng.normal(size=(10,168)).astype('float32');b=np.zeros(10,dtype='float32')
 return cache,y,w,b


def test_reset_each_recovers_stateless_predictions():
 c,y,w,b=fixture();order=np.arange(len(y));baseline=run_stream(c,y,order,w,b,'none')
 for policy in ['magnitude','pseudo','supervised']:
  out=run_stream(c,y,order,w,b,policy,intervention='reset_each')
  assert out['accuracy']==baseline['accuracy']
  assert out['pair_accuracy']==baseline['pair_accuracy']


def test_label_free_policies_ignore_supervised_feedback():
 c,y,w,b=fixture();order=np.arange(len(y));altered={**c,'supervised':np.zeros_like(c['supervised'])}
 for policy in ['none','magnitude','pseudo']:
  assert run_stream(c,y,order,w,b,policy)==run_stream(altered,y,order,w,b,policy)


def test_stream_contains_every_image_once():
 y=np.arange(1010)%10
 for kind in ['blocked','mixed']:
  order=stream_order(y,101,kind)
  assert np.array_equal(np.sort(order),np.arange(len(y)))
