Part-1:
main-pos-simple:
6 epochs
=== Training: ===
Fine 01 | train 1.29512 / 68.91% | dev 0.22636 / 93.66% | lr 1.5e-02
Fine 02 | train 0.16905 / 95.07% | dev 0.16867 / 94.84% | lr 1.5e-02
Fine 03 | train 0.14065 / 95.84% | dev 0.15871 / 95.09% | lr 1.5e-02
Fine 04 | train 0.13264 / 96.07% | dev 0.15406 / 95.31% | lr 1.5e-02
Learning-rate decayed → 6.0e-03; weight-decay scaled × 0.400
Fine 05 | train 0.12731 / 96.22% | dev 0.15503 / 95.28% | lr 6.0e-03
Fine 06 | train 0.09487 / 97.16% | dev 0.12996 / 95.85% | lr 6.0e-03


main-ner-simple:
epochs: 11
=== Training: ===
Fine 01 | train 8.37042 /  4.74% | dev 4.52137 /  6.15% | lr 3.4e-01
Fine 02 | train 3.12841 / 17.02% | dev 1.91502 / 38.87% | lr 3.4e-01
Fine 03 | train 1.64513 / 51.17% | dev 1.65751 / 51.37% | lr 3.4e-01
Fine 04 | train 1.47006 / 57.84% | dev 1.57562 / 55.61% | lr 3.4e-01
Fine 05 | train 2.30947 / 40.76% | dev 1.99793 / 43.76% | lr 3.4e-01
Learning-rate decayed → 1.0e-01; weight-decay scaled × 0.300
Fine 06 | train 1.67474 / 51.45% | dev 1.65007 / 50.40% | lr 1.0e-01
Fine 07 | train 0.84240 / 76.28% | dev 0.97793 / 71.63% | lr 1.0e-01
Fine 08 | train 0.67096 / 81.32% | dev 0.91363 / 73.62% | lr 1.0e-01
Fine 09 | train 0.61847 / 82.86% | dev 0.91494 / 73.25% | lr 1.0e-01
Learning-rate decayed → 3.1e-02; weight-decay scaled × 0.300
Fine 10 | train 0.60708 / 82.93% | dev 0.94545 / 72.76% | lr 3.1e-02
Fine 11 | train 0.34551 / 90.60% | dev 0.70003 / 80.03% | lr 3.1e-02


main-pos-simple-pretrained:
epochs: 10
=== Training: ===
Fine 01 | train 0.88532 / 76.17% | dev 0.24983 / 92.55% | lr 1.5e-02
Fine 02 | train 0.19433 / 94.26% | dev 0.17493 / 94.77% | lr 1.5e-02
Fine 03 | train 0.15365 / 95.53% | dev 0.16085 / 95.17% | lr 1.5e-02
Fine 04 | train 0.14193 / 95.89% | dev 0.15722 / 95.31% | lr 1.5e-02
Learning-rate decayed → 6.0e-03; weight-decay scaled × 0.400
Fine 05 | train 0.13654 / 96.02% | dev 0.15777 / 95.26% | lr 6.0e-03
Fine 06 | train 0.10274 / 96.99% | dev 0.13065 / 95.87% | lr 6.0e-03
Fine 07 | train 0.07712 / 97.66% | dev 0.12892 / 95.87% | lr 6.0e-03
Fine 08 | train 0.07427 / 97.79% | dev 0.12859 / 95.81% | lr 6.0e-03
Learning-rate decayed → 2.4e-03; weight-decay scaled × 0.400
Fine 09 | train 0.07213 / 97.82% | dev 0.12862 / 95.92% | lr 2.4e-03
Fine 10 | train 0.06128 / 98.13% | dev 0.12537 / 95.96% | lr 2.4e-03

main-ner-simple-pretrained:
epochs: 21
=== Training: ===
Fine 01 | train 2.86497 /  6.63% | dev 2.08342 / 27.32% | lr 3.0e-02
Fine 02 | train 2.22927 / 30.66% | dev 1.66992 / 47.53% | lr 3.0e-02
Fine 03 | train 1.51189 / 52.62% | dev 1.36176 / 58.62% | lr 3.0e-02
Fine 04 | train 1.07251 / 68.09% | dev 1.18629 / 63.99% | lr 3.0e-02
Fine 05 | train 0.79689 / 77.43% | dev 0.98476 / 70.65% | lr 3.0e-02
Fine 06 | train 0.61134 / 83.61% | dev 0.87432 / 74.57% | lr 3.0e-02
Fine 07 | train 0.51871 / 87.23% | dev 0.82768 / 75.47% | lr 3.0e-02
Fine 08 | train 0.46047 / 88.51% | dev 0.80818 / 75.81% | lr 3.0e-02
Fine 09 | train 0.41501 / 89.51% | dev 0.78730 / 76.43% | lr 3.0e-02
Fine 10 | train 0.39214 / 90.19% | dev 0.78604 / 76.33% | lr 3.0e-02
Fine 11 | train 0.37756 / 90.46% | dev 0.78191 / 76.32% | lr 3.0e-02
Fine 12 | train 0.35708 / 91.17% | dev 0.79232 / 75.75% | lr 3.0e-02
Learning-rate decayed → 3.0e-03; weight-decay scaled × 0.100
Fine 13 | train 0.34904 / 91.42% | dev 0.79213 / 76.19% | lr 3.0e-03
Fine 14 | train 0.25472 / 94.71% | dev 0.76461 / 77.11% | lr 3.0e-03
Fine 15 | train 0.19695 / 96.58% | dev 0.72473 / 78.24% | lr 3.0e-03
Fine 16 | train 0.14569 / 97.80% | dev 0.71247 / 78.96% | lr 3.0e-03
Fine 17 | train 0.11295 / 98.38% | dev 0.71353 / 79.03% | lr 3.0e-03
Fine 18 | train 0.09261 / 98.69% | dev 0.71224 / 79.38% | lr 3.0e-03
Fine 19 | train 0.07929 / 98.95% | dev 0.71421 / 79.49% | lr 3.0e-03
Learning-rate decayed → 3.0e-04; weight-decay scaled × 0.100
Fine 20 | train 0.07101 / 99.09% | dev 0.72003 / 79.62% | lr 3.0e-04
Fine 21 | train 0.06481 / 99.19% | dev 0.72083 / 79.64% | lr 3.0e-04


main-pos-subword:
epochs: 27
=== Training: ===
Fine 01 | train 1.75369 / 54.40% | dev 0.89486 / 74.07% | lr 1.0e-02
Fine 02 | train 0.72552 / 78.44% | dev 0.49637 / 85.09% | lr 1.0e-02
Fine 03 | train 0.46925 / 85.88% | dev 0.35840 / 89.20% | lr 1.0e-02
Fine 04 | train 0.35075 / 89.52% | dev 0.28275 / 91.50% | lr 1.0e-02
Fine 05 | train 0.28056 / 91.67% | dev 0.23582 / 92.82% | lr 1.0e-02
Fine 06 | train 0.23388 / 93.09% | dev 0.20461 / 93.80% | lr 1.0e-02
Fine 07 | train 0.20386 / 94.02% | dev 0.18567 / 94.23% | lr 1.0e-02
Fine 08 | train 0.18368 / 94.62% | dev 0.16998 / 94.74% | lr 1.0e-02
Fine 09 | train 0.17075 / 94.97% | dev 0.16064 / 94.99% | lr 1.0e-02
Fine 10 | train 0.16092 / 95.25% | dev 0.15316 / 95.26% | lr 1.0e-02
Fine 11 | train 0.15456 / 95.45% | dev 0.15079 / 95.37% | lr 1.0e-02
Fine 12 | train 0.14912 / 95.62% | dev 0.14619 / 95.51% | lr 1.0e-02
Fine 13 | train 0.14518 / 95.75% | dev 0.14497 / 95.42% | lr 1.0e-02
Fine 14 | train 0.14193 / 95.82% | dev 0.14469 / 95.49% | lr 1.0e-02
Fine 15 | train 0.13994 / 95.87% | dev 0.13791 / 95.81% | lr 1.0e-02
Fine 16 | train 0.13739 / 95.94% | dev 0.13518 / 95.88% | lr 1.0e-02
Fine 17 | train 0.13636 / 95.98% | dev 0.13403 / 95.85% | lr 1.0e-02
Fine 18 | train 0.13489 / 96.03% | dev 0.13497 / 95.91% | lr 1.0e-02
Learning-rate decayed → 1.0e-03; weight-decay scaled × 0.100
Fine 19 | train 0.13318 / 96.07% | dev 0.13472 / 95.81% | lr 1.0e-03
Fine 20 | train 0.11126 / 96.77% | dev 0.11621 / 96.32% | lr 1.0e-03
Fine 21 | train 0.09216 / 97.27% | dev 0.10922 / 96.46% | lr 1.0e-03
Fine 22 | train 0.08303 / 97.49% | dev 0.10575 / 96.49% | lr 1.0e-03
Fine 23 | train 0.07760 / 97.64% | dev 0.10376 / 96.62% | lr 1.0e-03
Fine 24 | train 0.07352 / 97.76% | dev 0.10298 / 96.61% | lr 1.0e-03
Fine 25 | train 0.07058 / 97.82% | dev 0.10246 / 96.58% | lr 1.0e-03
Fine 26 | train 0.06811 / 97.89% | dev 0.10142 / 96.67% | lr 1.0e-03
Fine 27 | train 0.06588 / 97.97% | dev 0.10104 / 96.70% | lr 1.0e-03

main-ner-subword: 25 epochs
=== Training: ===
Fine 01 | train 3.03276 /  9.27% | dev 2.86890 / 15.68% | lr 3.0e-02
Fine 02 | train 2.48030 / 25.08% | dev 2.06067 / 34.54% | lr 3.0e-02
Fine 03 | train 1.81212 / 42.81% | dev 1.48087 / 54.81% | lr 3.0e-02
Fine 04 | train 1.24642 / 62.14% | dev 1.10301 / 68.13% | lr 3.0e-02
Fine 05 | train 0.89001 / 73.86% | dev 0.91894 / 74.33% | lr 3.0e-02
Fine 06 | train 0.70155 / 79.97% | dev 0.88730 / 74.45% | lr 3.0e-02
Fine 07 | train 0.61267 / 82.58% | dev 0.83512 / 76.34% | lr 3.0e-02
Fine 08 | train 0.55356 / 84.22% | dev 0.77851 / 78.30% | lr 3.0e-02
Fine 09 | train 0.51771 / 85.53% | dev 0.75350 / 78.47% | lr 3.0e-02
Fine 10 | train 0.47720 / 86.48% | dev 0.76880 / 77.62% | lr 3.0e-02
Fine 11 | train 0.45008 / 87.03% | dev 0.70784 / 79.93% | lr 3.0e-02
Fine 12 | train 0.42630 / 87.87% | dev 0.71619 / 79.83% | lr 3.0e-02
Fine 13 | train 0.39917 / 88.65% | dev 0.69346 / 80.38% | lr 3.0e-02
Fine 14 | train 0.37748 / 89.23% | dev 0.71601 / 80.03% | lr 3.0e-02
Fine 15 | train 0.34893 / 90.31% | dev 0.68495 / 81.21% | lr 3.0e-02
Fine 16 | train 0.33848 / 90.54% | dev 0.67791 / 81.27% | lr 3.0e-02
Fine 17 | train 0.32414 / 90.83% | dev 0.70317 / 81.03% | lr 3.0e-02
Fine 18 | train 0.31294 / 91.09% | dev 0.65633 / 81.67% | lr 3.0e-02
Fine 19 | train 0.30754 / 91.21% | dev 0.71709 / 80.70% | lr 3.0e-02
Learning-rate decayed → 3.0e-03; weight-decay scaled × 0.100
Fine 20 | train 0.30026 / 91.59% | dev 0.71336 / 80.93% | lr 3.0e-03
Fine 21 | train 0.18711 / 95.45% | dev 0.65224 / 82.90% | lr 3.0e-03
Fine 22 | train 0.10427 / 98.03% | dev 0.62582 / 84.25% | lr 3.0e-03
Fine 23 | train 0.06985 / 98.88% | dev 0.62391 / 84.39% | lr 3.0e-03
Fine 24 | train 0.05243 / 99.31% | dev 0.64336 / 84.26% | lr 3.0e-03
Fine 25 | train 0.04297 / 99.47% | dev 0.64209 / 84.51% | lr 3.0e-03


main-pos-subword-pretrained:
epochs: 35
=== Training: ===
Fine 01 | train 1.65089 / 56.85% | dev 0.79987 / 76.42% | lr 1.0e-02
Fine 02 | train 0.66171 / 80.23% | dev 0.45972 / 86.23% | lr 1.0e-02
Fine 03 | train 0.44402 / 86.66% | dev 0.34324 / 89.61% | lr 1.0e-02
Fine 04 | train 0.34303 / 89.74% | dev 0.27474 / 91.70% | lr 1.0e-02
Fine 05 | train 0.27635 / 91.78% | dev 0.23185 / 92.86% | lr 1.0e-02
Fine 06 | train 0.23138 / 93.19% | dev 0.20156 / 93.94% | lr 1.0e-02
Fine 07 | train 0.20082 / 94.13% | dev 0.18204 / 94.44% | lr 1.0e-02
Fine 08 | train 0.18096 / 94.69% | dev 0.16730 / 94.92% | lr 1.0e-02
Fine 09 | train 0.16709 / 95.11% | dev 0.15821 / 95.12% | lr 1.0e-02
Fine 10 | train 0.15765 / 95.35% | dev 0.15062 / 95.43% | lr 1.0e-02
Fine 11 | train 0.15137 / 95.53% | dev 0.14537 / 95.65% | lr 1.0e-02
Fine 12 | train 0.14653 / 95.68% | dev 0.14140 / 95.64% | lr 1.0e-02
Fine 13 | train 0.14277 / 95.80% | dev 0.14066 / 95.61% | lr 1.0e-02
Fine 14 | train 0.13992 / 95.89% | dev 0.13870 / 95.75% | lr 1.0e-02
Fine 15 | train 0.13804 / 95.94% | dev 0.13647 / 95.85% | lr 1.0e-02
Fine 16 | train 0.13619 / 96.00% | dev 0.13561 / 95.91% | lr 1.0e-02
Fine 17 | train 0.13467 / 96.03% | dev 0.13467 / 95.89% | lr 1.0e-02
Fine 18 | train 0.13363 / 96.06% | dev 0.13518 / 95.69% | lr 1.0e-02
Fine 19 | train 0.13257 / 96.09% | dev 0.13402 / 95.81% | lr 1.0e-02
Fine 20 | train 0.13159 / 96.11% | dev 0.13406 / 95.86% | lr 1.0e-02
Fine 21 | train 0.13107 / 96.16% | dev 0.13196 / 95.99% | lr 1.0e-02
Fine 22 | train 0.13058 / 96.16% | dev 0.13208 / 95.94% | lr 1.0e-02
Learning-rate decayed → 1.0e-03; weight-decay scaled × 0.100
Fine 23 | train 0.12944 / 96.19% | dev 0.13366 / 95.90% | lr 1.0e-03
Fine 24 | train 0.10864 / 96.85% | dev 0.11611 / 96.37% | lr 1.0e-03
Fine 25 | train 0.09121 / 97.30% | dev 0.10941 / 96.46% | lr 1.0e-03
Fine 26 | train 0.08289 / 97.52% | dev 0.10652 / 96.54% | lr 1.0e-03
Fine 27 | train 0.07741 / 97.65% | dev 0.10475 / 96.59% | lr 1.0e-03
Fine 28 | train 0.07319 / 97.77% | dev 0.10320 / 96.62% | lr 1.0e-03
Fine 29 | train 0.07030 / 97.85% | dev 0.10217 / 96.67% | lr 1.0e-03
Fine 30 | train 0.06789 / 97.91% | dev 0.10177 / 96.62% | lr 1.0e-03
Fine 31 | train 0.06574 / 97.96% | dev 0.10139 / 96.69% | lr 1.0e-03
Fine 32 | train 0.06392 / 98.02% | dev 0.10139 / 96.69% | lr 1.0e-03
Fine 33 | train 0.06236 / 98.05% | dev 0.10114 / 96.68% | lr 1.0e-03
Fine 34 | train 0.06070 / 98.11% | dev 0.10115 / 96.65% | lr 1.0e-03
Learning-rate decayed → 1.0e-04; weight-decay scaled × 0.100
Fine 35 | train 0.05961 / 98.13% | dev 0.10118 / 96.71% | lr 1.0e-04

main-ner-subword-pretrained:
epochs: 14
=== Training: ===
Fine 01 | train 2.82726 / 15.93% | dev 2.17856 / 31.96% | lr 3.0e-02
Fine 02 | train 1.63924 / 49.25% | dev 1.19861 / 65.18% | lr 3.0e-02
Fine 03 | train 0.92857 / 72.85% | dev 0.92764 / 72.71% | lr 3.0e-02
Fine 04 | train 0.72086 / 79.13% | dev 0.83140 / 75.41% | lr 3.0e-02
Fine 05 | train 0.64813 / 81.32% | dev 0.82484 / 75.99% | lr 3.0e-02
Fine 06 | train 0.59902 / 82.96% | dev 0.83045 / 75.85% | lr 3.0e-02
Fine 07 | train 0.56800 / 83.69% | dev 0.79066 / 77.72% | lr 3.0e-02
Fine 08 | train 0.51881 / 85.08% | dev 0.77052 / 78.05% | lr 3.0e-02
Fine 09 | train 0.49445 / 85.65% | dev 0.74331 / 79.01% | lr 3.0e-02
Fine 10 | train 0.46783 / 86.50% | dev 0.75023 / 78.56% | lr 3.0e-02
Learning-rate decayed → 3.0e-03; weight-decay scaled × 0.100
Fine 11 | train 0.45076 / 87.07% | dev 0.74409 / 79.28% | lr 3.0e-03
Fine 12 | train 0.26026 / 92.89% | dev 0.62382 / 82.98% | lr 3.0e-03
Fine 13 | train 0.12019 / 97.37% | dev 0.62067 / 83.50% | lr 3.0e-03
Fine 14 | train 0.07570 / 98.65% | dev 0.62194 / 83.91% | lr 3.0e-03

main-pos-charcnn:
epochs: 11
=== Training: ===
Fine 01 | train 0.65927 / 82.70% | dev 0.19280 / 94.22% | lr 3.0e-02
Fine 02 | train 0.18478 / 94.56% | dev 0.17021 / 94.87% | lr 3.0e-02
Fine 03 | train 0.17263 / 94.88% | dev 0.16110 / 95.12% | lr 3.0e-02
Fine 04 | train 0.16654 / 95.05% | dev 0.15715 / 95.09% | lr 3.0e-02
Fine 05 | train 0.16306 / 95.12% | dev 0.15545 / 95.28% | lr 3.0e-02
Fine 06 | train 0.16077 / 95.20% | dev 0.16297 / 94.90% | lr 3.0e-02
Learning-rate decayed → 3.0e-03; weight-decay scaled × 0.100
Fine 07 | train 0.16000 / 95.23% | dev 0.16148 / 94.93% | lr 3.0e-03
Fine 08 | train 0.11432 / 96.45% | dev 0.10110 / 96.67% | lr 3.0e-03
Fine 09 | train 0.08190 / 97.33% | dev 0.09580 / 96.87% | lr 3.0e-03
Fine 10 | train 0.07115 / 97.66% | dev 0.09513 / 96.88% | lr 3.0e-03
Fine 11 | train 0.06460 / 97.87% | dev 0.09477 / 96.91% | lr 3.0e-03


main-ner-charcnn:
epochs: 20
=== Training: ===
Fine 01 | train 4.53503 /  1.09% | dev 4.32154 /  0.00% | lr 3.0e-02
Fine 02 | train 4.21265 /  0.01% | dev 3.85125 /  0.00% | lr 3.0e-02
Fine 03 | train 3.20753 /  2.00% | dev 2.27282 /  0.00% | lr 3.0e-02
Fine 04 | train 1.94087 / 19.93% | dev 1.71185 / 37.35% | lr 3.0e-02
Fine 05 | train 1.71198 / 34.74% | dev 1.50702 / 47.20% | lr 3.0e-02
Fine 06 | train 1.33875 / 52.73% | dev 1.07772 / 65.00% | lr 3.0e-02
Fine 07 | train 0.98992 / 69.39% | dev 0.80254 / 76.51% | lr 3.0e-02
Fine 08 | train 0.68851 / 80.79% | dev 0.62876 / 82.31% | lr 3.0e-02
Fine 09 | train 0.57336 / 83.88% | dev 0.61699 / 81.09% | lr 3.0e-02
Fine 10 | train 0.49234 / 86.59% | dev 0.53460 / 84.44% | lr 3.0e-02
Fine 11 | train 0.43500 / 88.44% | dev 0.52538 / 84.63% | lr 3.0e-02
Fine 12 | train 0.39415 / 89.79% | dev 0.49279 / 85.92% | lr 3.0e-02
Fine 13 | train 0.36280 / 90.73% | dev 0.48446 / 86.98% | lr 3.0e-02
Fine 14 | train 0.32837 / 91.74% | dev 0.47492 / 86.24% | lr 3.0e-02
Fine 15 | train 0.31250 / 92.30% | dev 0.49683 / 85.24% | lr 3.0e-02
Learning-rate decayed → 3.0e-03; weight-decay scaled × 0.100
Fine 16 | train 0.30034 / 92.48% | dev 0.48037 / 85.98% | lr 3.0e-03
Fine 17 | train 0.20032 / 96.06% | dev 0.42032 / 88.13% | lr 3.0e-03
Fine 18 | train 0.12865 / 97.65% | dev 0.39850 / 88.80% | lr 3.0e-03
Fine 19 | train 0.08495 / 98.63% | dev 0.39408 / 89.09% | lr 3.0e-03
Fine 20 | train 0.06267 / 99.13% | dev 0.39129 / 89.16% | lr 3.0e-03


5gram-cnn-2conv layers with residual:
epochs: 21
epoch 1: train-loss 1.86443 | val loss 1.6196 | perplexity 5.0510 | learning-rate 3.0e-03
epoch 2: train-loss 1.56051 | val loss 1.5366 | perplexity 4.6488 | learning-rate 3.0e-03
epoch 3: train-loss 1.50127 | val loss 1.4998 | perplexity 4.4807 | learning-rate 3.0e-03
epoch 4: train-loss 1.46981 | val loss 1.4857 | perplexity 4.4181 | learning-rate 3.0e-03
Best thee my shall be at:
If the raints?

CLADY:
Chere;
Unoble to the sovereign
his heir know'st to my lord?

EPHGNORTER:
Mark too hand, Seament by.
Gland-sence,
Against time!
Where of the Caplist to out t
epoch 5: train-loss 1.44770 | val loss 1.4668 | perplexity 4.3354 | learning-rate 3.0e-03
epoch 6: train-loss 1.47762 | val loss 1.4605 | perplexity 4.3080 | learning-rate 3.0e-03
epoch 7: train-loss 1.42063 | val loss 1.4503 | perplexity 4.2644 | learning-rate 3.0e-03
epoch 8: train-loss 1.40976 | val loss 1.4450 | perplexity 4.2421 | learning-rate 3.0e-03
epoch 9: train-loss 1.40027 | val loss 1.4411 | perplexity 4.2255 | learning-rate 3.0e-03
Best excelling.

First Serve were you things, all the loss,
I saw, sir?

Third now thou a sir, if it not girll in this time
Think
He that which he before hear,
The receiverse is. Prithee, lest thou pleased
epoch 10: train-loss 1.39315 | val loss 1.4355 | perplexity 4.2015 | learning-rate 3.0e-03
epoch 11: train-loss 1.43159 | val loss 1.4320 | perplexity 4.1870 | learning-rate 3.0e-03
epoch 12: train-loss 1.38156 | val loss 1.4288 | perplexity 4.1735 | learning-rate 3.0e-03
epoch 13: train-loss 1.37491 | val loss 1.4281 | perplexity 4.1707 | learning-rate 3.0e-03
epoch 14: train-loss 1.37032 | val loss 1.4282 | perplexity 4.1711 | learning-rate 3.0e-03
Best he sex your hearted Nerog sea
First Cupid't shunt themations his business.

Second Citizen:
Well, 'pardon.

GLOUCESTER:
Carrosciancertainment, for thinks, why -cure's eyes as him to a boy lays he chee
epoch 15: train-loss 1.36586 | val loss 1.4253 | perplexity 4.1590 | learning-rate 3.0e-03
epoch 16: train-loss 1.40762 | val loss 1.4225 | perplexity 4.1475 | learning-rate 3.0e-03
epoch 17: train-loss 1.35844 | val loss 1.4240 | perplexity 4.1537 | learning-rate 3.0e-03
Learning-rate decayed → 3.0e-04; weight-decay scaled × 0.100
epoch 18: train-loss 1.35440 | val loss 1.4238 | perplexity 4.1530 | learning-rate 3.0e-04
epoch 19: train-loss 1.30040 | val loss 1.3856 | perplexity 3.9971 | learning-rate 3.0e-04
Best of they us and pays.

BENVOLIO:
Not the noble hoxt
should be done took intented
To my moveables in me, unseveral that my slay
Thy father's to me in the fair Petruchious; and yet we have loathed you st
epoch 20: train-loss 1.28538 | val loss 1.3834 | perplexity 3.9883 | learning-rate 3.0e-04
epoch 21: train-loss 1.32710 | val loss 1.3809 | perplexity 3.9785 | learning-rate 3.0e-04