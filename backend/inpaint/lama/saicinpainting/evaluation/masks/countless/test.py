from copy import deepcopy

import numpy as np

import countless2d
import countless3d

def test_countless2d():
  def test_all_cases(fn, test_zero):
    case1 = np.array([ [ 1, 2 ], [ 3, 4 ] ]).reshape((2,2,1,1)) # all different
    case2 = np.array([ [ 1, 1 ], [ 2, 3 ] ]).reshape((2,2,1,1)) # two are same
    case1z = np.array([ [ 0, 1 ], [ 2, 3 ] ]).reshape((2,2,1,1)) # all different
    case2z = np.array([ [ 0, 0 ], [ 2, 3 ] ]).reshape((2,2,1,1)) # two are same
    case3 = np.array([ [ 1, 1 ], [ 2, 2 ] ]).reshape((2,2,1,1)) # two groups are same
    case4 = np.array([ [ 1, 2 ], [ 2, 2 ] ]).reshape((2,2,1,1)) # 3 are the same
    case5 = np.array([ [ 5, 5 ], [ 5, 5 ] ]).reshape((2,2,1,1)) # all are the same

    is_255_handled = np.array([ [ 255, 255 ], [ 1, 2 ] ], dtype=np.uint8).reshape((2,2,1,1))

    test = lambda case: fn(case)

    if test_zero:
      assert test(case1z) == [[[[3]]]] # d
      assert test(case2z) == [[[[0]]]] # a==b
    else:
      assert test(case1) == [[[[4]]]] # d
      assert test(case2) == [[[[1]]]] # a==b

    assert test(case3) == [[[[1]]]] # a==b
    assert test(case4) == [[[[2]]]] # b==c
    assert test(case5) == [[[[5]]]] # a==b
    
    assert test(is_255_handled) == [[[[255]]]] 

    assert fn(case1).dtype == case1.dtype

  test_all_cases(countless2d.simplest_countless, False)
  test_all_cases(countless2d.quick_countless, False)
  test_all_cases(countless2d.quickest_countless, False)
  test_all_cases(countless2d.stippled_countless, False)



  methods = [
    countless2d.zero_corrected_countless,
    countless2d.countless,
    countless2d.countless_if,
    # countless2d.counting, # counting doesn't respect order so harder to write a test
  ]

  for fn in methods:
    print(fn.__name__)
    test_all_cases(fn, True)

def test_stippled_countless2d():
  a = np.array([ [ 1, 2 ], [ 3, 4 ] ]).reshape((2,2,1,1)) 
  b = np.array([ [ 0, 2 ], [ 3, 4 ] ]).reshape((2,2,1,1)) 
  c = np.array([ [ 1, 0 ], [ 3, 4 ] ]).reshape((2,2,1,1)) 
  d = np.array([ [ 1, 2 ], [ 0, 4 ] ]).reshape((2,2,1,1)) 
  e = np.array([ [ 1, 2 ], [ 3, 0 ] ]).reshape((2,2,1,1)) 
  f = np.array([ [ 0, 0 ], [ 3, 4 ] ]).reshape((2,2,1,1)) 
  g = np.array([ [ 0, 2 ], [ 0, 4 ] ]).reshape((2,2,1,1)) 
  h = np.array([ [ 0, 2 ], [ 3, 0 ] ]).reshape((2,2,1,1)) 
  i = np.array([ [ 1, 0 ], [ 0, 4 ] ]).reshape((2,2,1,1)) 
  j = np.array([ [ 1, 2 ], [ 0, 0 ] ]).reshape((2,2,1,1)) 
  k = np.array([ [ 1, 0 ], [ 3, 0 ] ]).reshape((2,2,1,1)) 
  l = np.array([ [ 1, 0 ], [ 0, 0 ] ]).reshape((2,2,1,1)) 
  m = np.array([ [ 0, 2 ], [ 0, 0 ] ]).reshape((2,2,1,1)) 
  n = np.array([ [ 0, 0 ], [ 3, 0 ] ]).reshape((2,2,1,1)) 
  o = np.array([ [ 0, 0 ], [ 0, 4 ] ]).reshape((2,2,1,1)) 
  z = np.array([ [ 0, 0 ], [ 0, 0 ] ]).reshape((2,2,1,1)) 

  test = countless2d.stippled_countless

  # Note: We only tested non-matching cases above,
  # cases f,g,h,i,j,k prove their duals work as well
  # b/c if two pixels are black, either one can be chosen
  # if they are different or the same.

  assert test(a) == [[[[4]]]] 
  assert test(b) == [[[[4]]]] 
  assert test(c) == [[[[4]]]] 
  assert test(d) == [[[[4]]]] 
  assert test(e) == [[[[1]]]] 
  assert test(f) == [[[[4]]]]  
  assert test(g) == [[[[4]]]]  
  assert test(h) == [[[[2]]]]  
  assert test(i) == [[[[4]]]] 
  assert test(j) == [[[[1]]]]  
  assert test(k) == [[[[1]]]]  
  assert test(l) == [[[[1]]]]  
  assert test(m) == [[[[2]]]]  
  assert test(n) == [[[[3]]]]  
  assert test(o) == [[[[4]]]]  
  assert test(z) == [[[[0]]]]  

  bc = np.array([ [ 0, 2 ], [ 2, 4 ] ]).reshape((2,2,1,1)) 
  bd = np.array([ [ 0, 2 ], [ 3, 2 ] ]).reshape((2,2,1,1)) 
  cd = np.array([ [ 0, 2 ], [ 3, 3 ] ]).reshape((2,2,1,1)) 
  
  assert test(bc) == [[[[2]]]]
  assert test(bd) == [[[[2]]]]
  assert test(cd) == [[[[3]]]]

  ab = np.array([ [ 1, 1 ], [ 0, 4 ] ]).reshape((2,2,1,1)) 
  ac = np.array([ [ 1, 2 ], [ 1, 0 ] ]).reshape((2,2,1,1)) 
  ad = np.array([ [ 1, 0 ], [ 3, 1 ] ]).reshape((2,2,1,1)) 

  assert test(ab) == [[[[1]]]]
  assert test(ac) == [[[[1]]]]
  assert test(ad) == [[[[1]]]]
    
def test_countless3d():
  def test_all_cases(fn):
    alldifferent = [
      [
        [1,2],
        [3,4],
      ],
      [
        [5,6],
        [7,8]
      ]
    ]
    allsame = [
      [
        [1,1],
        [1,1],
      ],
      [
        [1,1],
        [1,1]
      ]
    ]

    assert fn(np.array(alldifferent)) == [[[8]]]
    assert fn(np.array(allsame)) == [[[1]]]

    twosame = deepcopy(alldifferent)
    twosame[1][1][0] = 2

    assert fn(np.array(twosame)) == [[[2]]]

    threemixed = [
      [
        [3,3],
        [1,2],
      ],
      [
        [2,4],
        [4,3]
      ]
    ]
    assert fn(np.array(threemixed)) == [[[3]]]

    foursame = [
      [
        [4,4],
        [1,2],
      ],
      [
        [2,4],
        [4,3]
      ]
    ]

    assert fn(np.array(foursame)) == [[[4]]]

    fivesame = [
      [
        [5,4],
        [5,5],
      ],
      [
        [2,4],
        [5,5]
      ]
    ]

    assert fn(np.array(fivesame)) == [[[5]]]

  def countless3d_generalized(img):
    return countless3d.countless_generalized(img, (2, 2, 2))
  def countless3d_dynamic_generalized(img):
    return countless3d.dynamic_countless_generalized(img, (2, 2, 2))

  methods = [
    countless3d.countless3d,
    countless3d.dynamic_countless3d,
    countless3d_generalized,
    countless3d_dynamic_generalized,
  ]

  for fn in methods:
    test_all_cases(fn)