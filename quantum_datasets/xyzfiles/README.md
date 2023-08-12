# XYZ File Format

We use a slightly modified version of `xyz` file format for the data generation pipeline to have sufficient information. For example,

```
3
H3+ 1
CCSD(T)/aug-cc-pVQZ
0.8737
H 0.0 0.5045 0.0
H 0.4369 -0.2522 0.0
H -0.4369 -0.2522 0.0
```

More generally, for any molecule we store the following:

```
<num_atoms>
<formula> <charge>
<basis-set>
<ground-state-bond-length>
<atom-1> <coordinates>
<atom-2> <coordinates>
...
<atom-3> <coordinates>
```

