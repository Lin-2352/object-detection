# Data

Everything under `data/` is git-ignored. Raw downloads live in `data/raw/`, the built
dataset in `data/processed/unified.npz` (about 750 MB, uint8, rebuilt by
`python -c "from hcr.unified import load_unified; load_unified(rebuild=True)"`).

## Sources and licences

| Source | Used for | Licence / citation |
|---|---|---|
| EMNIST byclass (NIST Special Database 19), `data/raw/emnist/gzip.zip` from <https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip> | handwritten 0-9, A-Z, a-z (62 classes) | NIST data, public domain / freely redistributable. Cohen, Afshar, Tapson, van Schaik (2017), *EMNIST: an extension of MNIST to handwritten letters*, arXiv:1702.05373 |
| HASYv2, `data/raw/hasy/HASYv2.tar.bz2` from <https://zenodo.org/records/259444> (download by hand, no code fetches it) | handwritten keyboard symbols (17 of 32, see below) | CC-BY-SA 4.0. Thoma (2017), *The HASYv2 dataset*, arXiv:1701.08380 |
| Rendered system fonts (Pillow, `C:\Windows\Fonts`) | printed glyphs for all 94 classes | Generated on the fly; no font files or renders are redistributed |

The processed `unified.npz` contains derived HASYv2 images, so redistributing it carries
the HASYv2 CC-BY-SA 4.0 share-alike and attribution terms.

## Honesty note: fonts are printed, not handwritten

Source 2 is *printed* text, augmented (rotation +-15 deg, shear, scale, stroke
dilation/erosion, blur, noise) but still machine-drawn. It supplies 104266 of
742742 training images, and the 15 symbols HASYv2 lacks are
**printed-only**. Handwritten accuracy on those 15 classes cannot be inferred from the test set
and should be treated as unknown. Because every image is cropped to its ink box, scaled to 20 px
and centred (the EMNIST convention), size/position cues vanish: `.` `,` `'` `` ` `` `"` `-` `_`
look alike as blobs/strokes, and `'`/`,`/`.` are genuinely confusable in this representation.

## HASYv2 coverage of the 32 keyboard symbols

Only true visual matches are mapped (checked against HASYv2 `symbols.csv`; the loader also
asserts id and LaTeX name agree). Covered: 17/32.

| char | HASYv2 id | LaTeX name |
|---|---|---|
| `#` | 266 | `\#` |
| `$` | 254 | `\$` |
| `%` | 267 | `\%` |
| `&` | 265 | `\&` |
| `*` | 514 | `\ast` |
| `-` | 195 | `-` |
| `+` | 196 | `+` |
| `[` | 918 | `[` |
| `]` | 919 | `]` |
| `{` | 257 | `\{` |
| `}` | 261 | `\}` |
| `\` | 949 | `\backslash` |
| `\|` | 920 | `\|` |
| `<` | 187 | `<` |
| `>` | 188 | `>` |
| `/` | 922 | `/` |
| `~` | 618 | `\sim` |

Missing (no HASYv2 class): `!` `@` `(` `)` `_` `=` `;` `:` `'` `"` `,` `.` `?` `^` ```. Rejected look-alikes: `\wedge` (not `^`),
`\cdot` (not `.`), `\prime` (not `'`), `\|` (double bar), `\mid` (duplicate of `|`).

## Unified dataset (`hcr.unified.load_unified`)

94 classes in ASCII order. Sources: E = EMNIST, H = HASYv2, F = fonts.

* **EMNIST**: official train/test kept; val = stratified 10% of train (no writer ids exist,
  so val shares writers with train).
* **HASYv2**: split by writer id (test ~20% of writers, val ~10% of the rest). One aggregate id
  (16925) holds about 89% of the mapped samples and cannot be split, so it is pinned to train.
  Consequence: HASY val/test are small and writer-diverse rather than large.
* **Fonts**: split by font file (229 train / 25 val / 63 test fonts); no font *file* appears in two splits. This is weaker than family-disjoint: sibling styles can be on different sides (`arial.ttf` and `arialbi.ttf` are train, `arialbd.ttf` and `ariali.ttf` are test). A review comparing raw glyphs found that 13 of the 88 val/test fonts have at least half of their glyphs almost identical to a training font, so printed-font test accuracy is somewhat optimistic (that is 5.7 % of the test images).
* **Cache**: `data/processed/unified.npz` is keyed only on its code version. Changing `font_dir`, `hasy_dir`, the installed fonts or a parameter does not rebuild it, and `load_unified(rebuild=True)` overwrites it. Delete it when those change.
* **Known data-quality issues** (from the review, not fixed): about 3 % of printed training images are almost blank after the erode/blur augmentation (worst for the backslash, slash, pipe, underscore, equals and backtick classes), thin brackets and pipes can become a slanted stroke that looks like a slash or backslash, and for `+ - < >` printed fonts are about 97 % of the training images because HASYv2 has only about 70-90 of each.
* Exact-pixel duplicates of train found in val/test are removed (2 val, 11 test).
* **Balancing**: alphanumerics get 300 printed train samples on top of EMNIST; every symbol
  class is topped up with font renders to at least 3000 train samples including its HASY samples.
  Val gets 10% and test 75 font renders per class from held-out fonts. EMNIST's own class
  imbalance (e.g. rare letters) is not altered.

Split sizes (E/H/F): train 628142/10334/104266 = 742742, val 69788/45/10419 = 80252,
test 116318/165/7050 = 123533.

### Per-class counts (E/H/F)

| class | train E/H/F | val E/H/F | test E/H/F |
|---|---|---|---|
| `!` | 0/0/3000 | 0/0/300 | 0/0/75 |
| `"` | 0/0/3000 | 0/0/300 | 0/0/75 |
| `#` | 0/1067/1933 | 0/0/193 | 0/0/75 |
| `$` | 0/378/2622 | 0/2/262 | 0/7/75 |
| `%` | 0/881/2119 | 0/0/211 | 0/6/75 |
| `&` | 0/1278/1722 | 0/0/172 | 0/8/75 |
| `'` | 0/0/3000 | 0/0/300 | 0/0/75 |
| `(` | 0/0/3000 | 0/0/300 | 0/0/75 |
| `)` | 0/0/3000 | 0/0/300 | 0/0/75 |
| `*` | 0/546/2454 | 0/0/245 | 0/11/75 |
| `+` | 0/69/2931 | 0/5/293 | 0/16/75 |
| `,` | 0/0/3000 | 0/0/300 | 0/0/75 |
| `-` | 0/88/2912 | 0/5/291 | 0/20/75 |
| `.` | 0/0/3000 | 0/0/300 | 0/0/75 |
| `/` | 0/524/2476 | 0/3/247 | 0/5/75 |
| `0` | 31127/0/300 | 3458/0/30 | 5777/0/75 |
| `1` | 34537/0/300 | 3837/0/30 | 6330/0/75 |
| `2` | 30783/0/300 | 3420/0/30 | 5869/0/75 |
| `3` | 31629/0/300 | 3514/0/30 | 5969/0/75 |
| `4` | 30181/0/300 | 3354/0/30 | 5619/0/75 |
| `5` | 28274/0/300 | 3142/0/30 | 5190/0/75 |
| `6` | 30809/0/300 | 3423/0/30 | 5705/0/75 |
| `7` | 32179/0/300 | 3575/0/30 | 6139/0/75 |
| `8` | 30551/0/300 | 3395/0/30 | 5633/0/75 |
| `9` | 30462/0/300 | 3385/0/30 | 5685/0/75 |
| `:` | 0/0/3000 | 0/0/300 | 0/0/75 |
| `;` | 0/0/3000 | 0/0/300 | 0/0/75 |
| `<` | 0/85/2915 | 0/5/291 | 0/24/75 |
| `=` | 0/0/3000 | 0/0/300 | 0/0/75 |
| `>` | 0/80/2920 | 0/5/292 | 0/16/75 |
| `?` | 0/0/3000 | 0/0/300 | 0/0/75 |
| `@` | 0/0/3000 | 0/0/300 | 0/0/75 |
| `A` | 5766/0/300 | 641/0/30 | 1062/0/75 |
| `B` | 3490/0/300 | 388/0/30 | 648/0/75 |
| `C` | 9085/0/300 | 1009/0/30 | 1739/0/75 |
| `D` | 4106/0/300 | 456/0/30 | 779/0/75 |
| `E` | 4441/0/300 | 493/0/30 | 851/0/75 |
| `F` | 8264/0/300 | 918/0/30 | 1440/0/75 |
| `G` | 2265/0/300 | 252/0/30 | 447/0/75 |
| `H` | 2837/0/300 | 315/0/30 | 521/0/75 |
| `I` | 10751/0/300 | 1195/0/30 | 2048/0/75 |
| `J` | 3386/0/300 | 376/0/30 | 626/0/75 |
| `K` | 2221/0/300 | 247/0/30 | 382/0/75 |
| `L` | 4568/0/300 | 508/0/30 | 810/0/75 |
| `M` | 8102/0/300 | 900/0/30 | 1485/0/75 |
| `N` | 7413/0/300 | 824/0/30 | 1351/0/75 |
| `O` | 22485/0/300 | 2498/0/30 | 4156/0/75 |
| `P` | 7512/0/300 | 835/0/30 | 1397/0/75 |
| `Q` | 2345/0/300 | 260/0/30 | 413/0/75 |
| `R` | 4566/0/300 | 507/0/30 | 809/0/75 |
| `S` | 18688/0/300 | 2076/0/30 | 3508/0/75 |
| `T` | 8838/0/300 | 982/0/30 | 1576/0/75 |
| `U` | 11342/0/300 | 1260/0/30 | 2002/0/75 |
| `V` | 4173/0/300 | 464/0/30 | 796/0/75 |
| `W` | 4225/0/300 | 470/0/30 | 805/0/75 |
| `X` | 2494/0/300 | 277/0/30 | 432/0/75 |
| `Y` | 4269/0/300 | 474/0/30 | 798/0/75 |
| `Z` | 2431/0/300 | 270/0/30 | 464/0/75 |
| `[` | 0/902/2098 | 0/2/209 | 0/12/75 |
| `\` | 0/259/2741 | 0/1/274 | 0/0/75 |
| `]` | 0/254/2746 | 0/2/274 | 0/3/75 |
| `^` | 0/0/3000 | 0/0/300 | 0/0/75 |
| `_` | 0/0/3000 | 0/0/300 | 0/0/75 |
| ``` | 0/0/3000 | 0/0/300 | 0/0/75 |
| `a` | 9030/0/300 | 1003/0/30 | 1644/0/75 |
| `b` | 4643/0/300 | 516/0/30 | 853/0/75 |
| `c` | 2569/0/300 | 285/0/30 | 431/0/75 |
| `d` | 9159/0/300 | 1018/0/30 | 1683/0/75 |
| `e` | 22168/0/300 | 2463/0/30 | 4092/0/75 |
| `f` | 2305/0/300 | 256/0/30 | 400/0/75 |
| `g` | 3318/0/300 | 369/0/30 | 589/0/75 |
| `h` | 7864/0/300 | 874/0/30 | 1479/0/75 |
| `i` | 2453/0/300 | 272/0/30 | 427/0/75 |
| `j` | 1706/0/300 | 190/0/30 | 317/0/75 |
| `k` | 2242/0/300 | 249/0/30 | 466/0/75 |
| `l` | 13786/0/300 | 1532/0/30 | 2535/0/75 |
| `m` | 2381/0/300 | 264/0/30 | 464/0/75 |
| `n` | 10276/0/300 | 1141/0/30 | 1898/0/75 |
| `o` | 2474/0/300 | 275/0/30 | 466/0/75 |
| `p` | 2203/0/300 | 245/0/30 | 368/0/75 |
| `q` | 2695/0/300 | 299/0/30 | 505/0/75 |
| `r` | 12695/0/300 | 1409/0/30 | 2319/0/75 |
| `s` | 2429/0/300 | 270/0/30 | 437/0/75 |
| `t` | 16436/0/300 | 1826/0/30 | 2965/0/75 |
| `u` | 2547/0/300 | 283/0/30 | 482/0/75 |
| `v` | 2619/0/300 | 291/0/30 | 468/0/75 |
| `w` | 2427/0/300 | 270/0/30 | 467/0/75 |
| `x` | 2540/0/300 | 282/0/30 | 470/0/75 |
| `y` | 2129/0/300 | 236/0/30 | 381/0/75 |
| `z` | 2453/0/300 | 272/0/30 | 451/0/75 |
| `{` | 0/1126/1874 | 0/2/187 | 0/8/75 |
| `\|` | 0/501/2499 | 0/2/249 | 0/7/75 |
| `}` | 0/313/2687 | 0/3/268 | 0/5/75 |
| `~` | 0/1983/1017 | 0/8/101 | 0/17/75 |
