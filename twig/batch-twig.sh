#!/bin/bash
n_first_epochs=5
n_second_epochs=10
kfold=0
test_mode="hyp"
tag="pretrain-4"

datasets="CoDExSmall-DBpedia50-Kinships-OpenEA"
./TWIG_pipeline.sh \
    $datasets \
    $n_first_epochs \
    $n_second_epochs \
    $tag \
    $kfold \
    $test_mode 

datasets="CoDExSmall-DBpedia50-Kinships-UMLS"
./TWIG_pipeline.sh \
    $datasets \
    $n_first_epochs \
    $n_second_epochs \
    $tag \
    $kfold \
    $test_mode 

datasets="CoDExSmall-DBpedia50-OpenEA-UMLS"
./TWIG_pipeline.sh \
    $datasets \
    $n_first_epochs \
    $n_second_epochs \
    $tag \
    $kfold \
    $test_mode 

datasets="CoDExSmall-Kinships-OpenEA-UMLS"
./TWIG_pipeline.sh \
    $datasets \
    $n_first_epochs \
    $n_second_epochs \
    $tag \
    $kfold \
    $test_mode 

datasets="DBpedia50-Kinships-OpenEA-UMLS"
./TWIG_pipeline.sh \
    $datasets \
    $n_first_epochs \
    $n_second_epochs \
    $tag \
    $kfold \
    $test_mode 
