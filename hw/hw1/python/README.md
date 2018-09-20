# Part 1 - extracting n-grams from a sentence

```python
# input_size = 3, n = 1
[ "natural", "language", "processsing" ]
=>
[ ('START',)
, ('natural',)
, ('language',)
, ('processing',)
, ('STOP', )
] # output_size = input_size + 2

# input_size = 3, n = 2
["natural","language","processing"]
=>
[ ('START', 'natural')
, ('natural', 'language')
, ('language', 'processing')
, ('processing', 'STOP')
] # output_size = input_size + 1

# input_size = 3, n = 3
["natural","language","processing"]
=>
[ ('START', 'START', 'natural')
, ('START', 'natural', 'language')
, ('natural', 'language', 'processing')
, ('language', 'processing', 'STOP')
] # output_size = input_size + 1

# input_size = 3, n = 4
["natural","language","processing"]
=>
[ ('START', 'START', 'START', 'natural')
, ('START', 'START', 'natural', 'language')
, ('START', 'natural', 'language', 'processing')
, ('natural', 'language', 'processing', 'STOP')
] # output_size = input_size + 1

# input_size = 3, n = 5
["natural","language","processing"]
=>
[ ('START', 'START', 'START', 'START', 'natural')
, ('START', 'START', 'START', 'natural', 'language')
, ('START', 'START', 'natural', 'language', 'processing')
, ('START', 'natural', 'language', 'processing', 'STOP')
] # output_size = input_size + 1
```

unigram (`n = 1`) is a special case where an extra `("START",)` at the beginning is needed

