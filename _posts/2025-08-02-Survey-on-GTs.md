---
layout: distill
title: "Structure-Preserving Graph Transformer 모델들에 대한 Survey 연구 회고"
description: Graph Transformer의 Design Space에 대한 고민들
date: 2025-08-02 00:00:00+0900
featured: true
tags:
  - GNNs
  - GTs
  - GRL
  - GraphML
  - Papers
categories:
  - Research
authors:
  - name: O-Joun Lee
    url: "https://nslab-cuk.github.io/member/ojlee"
    affiliations:
      name: The Catholic University of Korea
toc:
  - name: Motivation
  - name: Node Feature Modulation
  - name: Context Node Sampling
  - name: Graph Rewriting
  - name: Transformer Architecture Improvements
  - name: Conclusion
---

<br/>
<div style="display: block; margin-left: auto; margin-right: auto; width:100%; text-align:center;">
  <a href="https://arxiv.org/abs/2401.16176" class="btn btn--primary">read the paper</a> 
  <a href="https://ojs.aaai.org/index.php/AAAI/article/view/29138" class="btn btn--primary">read UGT paper</a> 
  <a href="https://ieeexplore.ieee.org/document/10974679" class="btn btn--primary">read CGT paper</a> 
</div> <br/>

이 글에서는 우리 연구실(Network Science Lab @ CUK)의 Van Thuy Hoang 박사과정과 함께 진행한 Structure-Preserving Graph Transformer 모델들에 대한 Survey 연구<d-footnote>Van Thuy Hoang, O-Joun Lee: A Survey on Structure-Preserving Graph Transformers. arXiv preprint 01/2024; arXiv:2401.16176. (Preprint)</d-footnote>의 배경과, 논문에 포함되지 못했던 고민들을 정리하고자 한다. 해당 논문은 IJCAI 2024의 Survey Track에 투고하였으나 아쉽게도 채택되지는 못했다.

# Motivation

Van Thuy Hoang 박사과정과 함께 [UGT](https://ojs.aaai.org/index.php/AAAI/article/view/29138)와 [CGT](https://ieeexplore.ieee.org/document/10974679)에 대한 연구를 진행하면서, Structure-Preserving Graph Transformers의 설계원칙과 Design Space에 대한 우리의 이해를 어느정도 정립할 수 있었다. 
앞선 UGT와 CGT 연구를 진행하면서 늘 고민했던 부분은 “Graph Transformer는 그래프의 구조를 얼마나 잘 보존할 수 있을까?”라는 질문이었다. 
Multi-head Attention을 바탕으로 전체 입력 토큰 간의 정보를 전파하는 Transformer는 Permutation Invariance를 보장할 수 있지만, Over-globalization에 대한 문제를 피하기 힘들다. 
Graph Transformer 모델들이 지역적/전역적 구조 정보를 최대한 보존하여, 지역적 인접성을 갖거나 구조적 유사성을 갖는 노드들 만이 비슷한 벡터 표현을 갖도록 하는 Structure-Preserving Graph Transformer 모델들에 대한 고민이 필요한 이유이다.
우리는 이를 위한 Graph Transformers의 구조 정보를 주입하기 위한 개선 방향성을 Node Feature Modulation, Context Node Sampling, Graph Rewriting (혹은 Rewiring), Transformer Architecture Improvements의 네 가지로 분류하고 여기에 대한 우리의 생각을 Short Survey 형태로 정리했다.


## Node Feature Modulation  

첫 번째 축은 구조 정보를 초기 노드 특징 벡터에 주입하는 방식으로 자연어처리 분야에서 토큰의 순서나 문장 구분을  Positional encoding을 통해 전달하는 것과 비슷한 접근이다.
대표적인 방법은 Laplacian eigenvector를 Positional encoding으로 사용하는 방식이다. 
Laplacian eigenvector는 1-hop connectivity에 대한 정보만을 내포한다는 한계을 가짐에도, 많은 연구에서 분자 성질 예측 벤치마크 상에서 SOTA 성능을 보고한 바 있다.
특히, TokenGT의 경우 encoding을 엣지 토큰까지 확장하면 2-WL 수준의 구조 식별력을 달성할 수 있고 구조 정보가 모델 표현력에도 기여한다는 것을 보였다.

Laplacian eigenvector 보다 큰 규모의 구조 정보를 나타낼 수 있는 Random walk, Transition probability, Degree sequence, Shortest path distance (SPD) 기반의 방법들 또한 자주 사용된다. 
SPD의 경우에는 전역적 구조 상에서 목표 노드의 상대적 위치를 표현하는데 장점이 있으며, 나머지 특징들은 Multi-scale로 목표 노드 주변의 구조를 표현하는데 장점을 갖는다.
특히, 보고된 실험 결과들에 따르면 Heterophily 특성이 강한 그래프에서는 단순한 Eigenvector positional encoding보다는 Random walk 기반의 Positional encoding이 더 일관된 예측을 보인다.
이외에도 Graph kernel로 많이 활용되는 다양한 구조적 특징들이 Positional encoding, Initial residual, Attention score 등을 통해 모델에 추가적인 입력으로 제공되거나, Graph augmentation이나 Node sampling에 활용된다.


## Context Node Sampling  

두 번째 축은 Graph Transformer의 Receptive filed, 즉, 목표 노드의 Context 노드 집합을 어떻게 정의할지에 대한 문제다. 
모든 입력 토큰 사이에 정보를 전파하는 Transformer는 노드 간 인접성 정보를 훼손할 수 있으며, 일반적으로 1-hop 이웃을 Context의 범위로 하는 Graph Neural Network 모델들은 Long-range Dependecy 문제와 여기서 이어지는 Over-smoothing이나 Over-squashing 문제들을 마주하기 마련이다. 
이 문제를 해소하기 위해서는 전체 노드와 1-hop 이웃 사이의 어느 지점으로 Context 노드의 범위를 조정할 필요가 있으며, 이때 가장 크게 고려되어야 할 사항은 어떤 노드들이 충분히 비슷한 노드들인가 그리고 어떤 노드들에게 비슷한 벡터 표현이 주어져야 할지이다.

지역적인 샘플링을 사용하는 Graphformer나 EGT, Gophormer 같은 모델들은 주로 k-hop 이웃이나 ego-network를 활용해 Context의 범위를 제한한다. 
반면 DGT는 노드 특징 유사도를, UGT는 노드의 구조적 유사도를 기반으로 하는 전역적인 Context 노드 샘플링을 함께 이용하는 방식을 취한다. 
보고된 실험 결과들을 살펴보면 Heterophily 특성이 강한 그래프에서는 구조적 유사도 기반 샘플링이 노드 특징 유사도 기반의 방식보다 더 안정적이고 일반화 성능을 보였다. 
이런 결과들은 구조적으로 의미 있는 정보는 지역적 연결성만이 아니라 전역적인 Context에서 비롯된다는 점을 뒷받침한다.


## Graph Rewriting  

세 번째 축은 그래프 자체를 재구성하거나 변형하여 구조 특징을 반영하고 Attention score의 계산 범위를 조정하는 방식이다. 
노드 샘플링을 바탕으로 하는 방법들이 목표 노드와 연결성이나 구조적 유사성을 갖는 노드들을 Context 노드 집합에 포함시키는 것과 달리, 그래프 재구성의 경우 이런 노드들을 목표 노드와 연결하는 (Virtual) 엣지를 추가한다.

GT나 SAN처럼 Context 노드 집합 내의 모든 노드를 완전히 연결하는 방식은 작은 그래프에서는 유효하지만, 노드 수가 수만 개 단위에 이르면 메모리 비용이 지나치게 증가한다.
그래프 Coarsening을 도입한 Coarformer나 ANS‑GT는 Super 노드를 이용해 복잡도를 줄이지만, 이 역시 그래프 Partitioning 방법과 데이터 특성에 따라 성능 차이가 크다. 
이에 비해, UGT에서 채택한 Virtual 엣지 추가는 구조적으로 유사한 노드 쌍을 연결하는 새로운 엣지를 추가하는 간단한 방식만으로도 Heterophily 환경에서 유의미한 성능 향상을 얻을 수 있다는 장점이 있다.
하지만, 추가할 Virtual 엣지의 수를 유동적으로 조절하지 못한다는 것이 실제 그래프의 구조에 따라서는 Noisy Signal 문제를 일으킴이 실험적으로 활용되기도 했다.
이런 문제들을 해결하기 위한 Adaptive sparsification에 대한 연구가 필요한 이유이다.


## Architecture Improvements  

네 번째 축은 Transformer 자체의 구조 및 Attention Mechanism을 개선하여 구조 정보를 학습하도록 하는 방향이다. 
구조 식별력은 구조적으로 다른 노드 쌍 혹은 그래프 쌍에 대해 식별 가능한 벡터 표현을 부여하느냐에 대한 문제로, GIN 등의 널리 알려진 모델들을 통해 논의된 바와 같이 모델 자체가 구조와 벡터 표현을 1:1 맵핑할 수 있을 정도의 표현력을 갖느냐하는 문제가 가장 주요하다.
하지만, UGT와 CGT에서 이용된 것처럼, Initial residual connection을 통해 구조적 특징 벡터를 전달하거나 Attention score가 구조적 유사도를 반영할 수 있도록 한다면, 모델 Efficiency와 구조 표현력을 동시에 확보하는데 도움이 된다.

Attention score를 보강하는데는 구조적 유사도에 대한 항을 추가해주거나 노드 특징 행렬에 구조적 특징을 추가하는 방식이 많이 활용된다. 구조적 특징으로는 Node Feature Modulation의 경우와 같이 Graph Kernel에서 부터 이용되던 다양한 구조 정보들이 활용된다. UGT와 CGT는 타겟 노드의 주변 구조에 대한 Description 능력이 좋은 Degree Sequence나 SPD 같은 정보들을 High-order 인접성 정보를 잘 나타낼 수 있는 Transition probability 등과 함께 활용하였는데, 이는 구조적 유사도만 이용할 경우 Attention이 그래프 전역에 분산될 수 있으므로 High-order 인접성을 바탕으로 더 의미있는 Sparse attention을 형성하고자 하는 목적을 가지고 있다.

Graph Transformer에 대한 비교적 초창기 연구들 중에는 Transformer 레이어 이전에 GNN 레이어를 쌓아 전역적인 Self-attention 적용으로 인한 구조 정보 손실을 보완하려는 시도들이 있었다. Transformer와 GNN 구조 간의 Context 범위 및 정보전파 범위 차이는 둘이 병용될 경우 Over-globalization과 Long-range Dependency 문제 등을 어느정도 완화하지만, 근본적인 해결책이 되기에는 어려움이 있다.


# Conclusion

CGT에 대한 연구를 마무리하고 IJCAI 2024에 투고하기 위한 작업을 하면서, 막간의 여유를 이용해 이 Short Survey 또한 IJCAI 2024의 Survey Track에 투고하기 위해 준비했다. 
두 논문 모두 좋은 성과를 거두지는 못한 지금 되돌아보자면, CGT에 대한 논문을 더 다듬는데 시간과 에너지를 더 투자하고 Graph Transformer의 Design space에 대한 논의는 조금더 시간을 두고 우리의 생각을 다듬었다면 좋은 성과가 있지 않았을까 생각한다.

이 Short Survey 작업은 Structure preservation 관점에서 Graph Transformer의 Design space에 대한 단순한 정리를 넘어, 각 설계 요소들이 서로 어떻게 영향을 주고받는지를 보다 체계적으로 조망할 수 있었다. 네 가지 축은 각각 모델의 표현력, 확장성, 일반화 능력에 영향을 주며, Graph Transformer를 설계하는 데 있어 반드시 함께 고려되어야 할 요소들임을 다시금 확인할 수 있었다.


