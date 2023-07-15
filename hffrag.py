import uproot
import awkward
import numpy
import matplotlib.figure as figure
import matplotlib.pyplot as plt
import keras
import keras.layers as layers
from Sum import Sum


MASKVAL = -999
MAXTRACKS = 8
BATCHSIZE = 64
EPOCHS = 250
MAXEVENTS = 99999999999999999
# VALFACTOR = 10
LR = 1e-3

# this opens up the root file and points to the "CharmAnalysis" tree inside.
tree = uproot.open("hffrag.root:CharmAnalysis")


# decide which branches of the tree we actually want to look at
# uncomment as needed!
# not currently used!
branches = \
  [ \

  # true jet information
    "AnalysisAntiKt4TruthJets_pt"
  # , "AnalysisAntiKt4TruthJets_eta"
  # , "AnalysisAntiKt4TruthJets_phi"
  # , "AnalysisAntiKt4TruthJets_m"


  # true b-hadron information
  # these b-hadrons are inside the truth jets
  , "AnalysisAntiKt4TruthJets_ghostB_pdgId"
  , "AnalysisAntiKt4TruthJets_ghostB_pt"
  # , "AnalysisAntiKt4TruthJets_ghostB_eta"
  # , "AnalysisAntiKt4TruthJets_ghostB_phi"
  # , "AnalysisAntiKt4TruthJets_ghostB_m"
  

  # reconstructed jet information
  # , "AnalysisJets_pt_NOSYS"
  # , "AnalysisJets_eta"
  # , "AnalysisJets_phi"
  # , "AnalysisJets_m"


  # reconstructed track information
  , "AnalysisTracks_pt"
  , "AnalysisTracks_eta"
  , "AnalysisTracks_phi"
  , "AnalysisTracks_z0sinTheta"
  , "AnalysisTracks_d0sig"
  , "AnalysisTracks_d0"
  , "AnalysisTracks_d0sigPV"
  , "AnalysisTracks_d0PV"
  ]


# true jet information
jetfeatures = \
  [ "AnalysisAntiKt4TruthJets_pt"
  , "AnalysisAntiKt4TruthJets_eta"
  , "AnalysisAntiKt4TruthJets_phi"
  , "AnalysisAntiKt4TruthJets_ghostB_pt"
  ]

# true b-hadron information
# these b-hadrons are inside the truth jets
# bhadfeatures = \
#   [ "AnalysisAntiKt4TruthJets_ghostB_pt"
#   , "AnalysisAntiKt4TruthJets_ghostB_eta"
#   , "AnalysisAntiKt4TruthJets_ghostB_phi"
  # , "AnalysisAntiKt4TruthJets_ghostB_m"
  # ]
  

# reconstructed track information
trackfeatures = \
  [ "AnalysisTracks_pt"
  , "AnalysisTracks_eta"
  , "AnalysisTracks_phi"
  ]

# read in the requested branches from the file
features = tree.arrays(jetfeatures + trackfeatures, entry_stop=MAXEVENTS)


def matchTracks(jets, trks):
  jeteta = jets["AnalysisAntiKt4TruthJets_eta"]
  jetphi = jets["AnalysisAntiKt4TruthJets_phi"]

  trketas = trks["AnalysisTracks_eta"]
  trkphis = trks["AnalysisTracks_phi"]

  detas = jeteta - trketas
  dphis = numpy.abs(jetphi - trkphis)

  # deal with delta phis being annoying
  awkward.where(dphis > numpy.pi, dphis - numpy.pi, dphis)

  return numpy.sqrt(dphis**2 + detas**2) < 0.4


def ptetaphi2pxpypz(ptetaphi):
  pts = ptetaphi[:,0:1]
  etas = ptetaphi[:,1:2]
  phis = ptetaphi[:,2:3]

  pxs = pts * numpy.cos(phis)
  pys = pts * numpy.sin(phis)
  pzs = pts * numpy.sinh(etas)

  isinf = numpy.isinf(pzs)

  if numpy.any(isinf):
    print("inf from eta:")
    print(etas[isinf])
    raise ValueError("infinity from sinh(eta)")

  return numpy.concatenate([pxs, pys, pzs], axis=1)


def ptetaphi2pxpypz2(ptetaphi):
  pts = ptetaphi[:,:,0:1]
  etas = ptetaphi[:,:,1:2]
  phis = ptetaphi[:,:,2:3]

  mask = pts == MASKVAL
  pxs = numpy.where(mask, pts, pts * numpy.cos(phis))
  pys = numpy.where(mask, pts, pts * numpy.sin(phis))
  pzs = numpy.where(mask, pts, pts * numpy.sinh(etas))

  isinf = numpy.isinf(pzs)

  if numpy.any(isinf):
    print("inf from eta:")
    print(etas[isinf])
    raise ValueError("infinity from sinh(eta)")

  return numpy.concatenate([pxs, pys, pzs], axis=2)


# pads inputs with nans up to the given maxsize
def pad(xs, maxsize):
  ys = \
    awkward.fill_none \
    ( awkward.pad_none(xs, maxsize, axis=1, clip=True)
    , MASKVAL
    )[:,:maxsize]

  return awkward.to_regular(ys, axis=1)


def flatten1(xs, maxsize=-1):
  ys = {}
  for field in xs.fields:
    zs = xs[field]
    if maxsize > 0:
      zs = pad(zs, maxsize)
    ys[field] = zs

  return awkward.zip(ys)


events = \
  features[awkward.sum(features["AnalysisAntiKt4TruthJets_pt"] > 25000, axis=1) > 0]

jets = events[jetfeatures][:,0]

print("number of jets to train on:")
print(len(jets))
tracks = events[trackfeatures]

matchedtracks = tracks[matchTracks(jets, tracks)]
matchedtracks = flatten1(matchedtracks, MAXTRACKS)

bjets = awkward.sum(jets["AnalysisAntiKt4TruthJets_ghostB_pt"] > 5000, axis=1) > 0
jets = jets[bjets]
bhads = jets["AnalysisAntiKt4TruthJets_ghostB_pt"][:,0]
matchedtracks = matchedtracks[bjets]

from numpy.lib.recfunctions import structured_to_unstructured
jets = structured_to_unstructured(jets[jetfeatures[:-1]])
matchedtracks = structured_to_unstructured(matchedtracks)

jets = ptetaphi2pxpypz(jets).to_numpy()
tracks = ptetaphi2pxpypz2(matchedtracks.to_numpy())
bhads = bhads.to_numpy()


# returns a fixed set of bin edges
def fixedbinning(xmin, xmax, nbins):
  return numpy.mgrid[xmin:xmax:nbins*1j]


# define two functions to aid in plotting
def hist(xs, binning, normalized=False):
  ys = numpy.histogram(xs, bins=binning)[0]

  yerrs = numpy.sqrt(ys)

  if normalized:
    s = numpy.sum(ys)
    ys = ys / s
    yerrs = yerrs / s

  return ys, yerrs


def binneddensity(xs, binning, label=None, xlabel=None, ylabel="binned probability density"):
  fig = figure.Figure(figsize=(8, 8))
  plt = fig.add_subplot(111)

  ys , yerrs = hist(xs, binning, normalized=True)

  # determine the central value of each histogram bin
  # as well as the width of each bin
  # this assumes a fixed bin size.
  xs = (binning[1:]+binning[:-1]) / 2.0
  xerrs = ((binning[1:]-binning[:-1]) / 2.0)

  plt.errorbar \
    ( xs
    , ys
    , xerr=xerrs
    , yerr=yerrs
    , label=label
    , linewidth=0
    , elinewidth=2
    )

  plt.set_xlabel(xlabel)
  plt.set_ylabel(ylabel)

  return fig

def binneddensitysub(xs, binning, ax, label=None, xlabel=None, ylabel="binned probability density"):
  #fig = figure.Figure(figsize=(8, 8))
  #plt = fig.add_subplot(111)

  ys , yerrs = hist(xs, binning, normalized=True)

  # determine the central value of each histogram bin
  # as well as the width of each bin
  # this assumes a fixed bin size.
  xs = (binning[1:]+binning[:-1]) / 2.0
  xerrs = ((binning[1:]-binning[:-1]) / 2.0)

  ax.errorbar \
    ( xs
    , ys
    , xerr=xerrs
    , yerr=yerrs
    , label=label
    , linewidth=0
    , elinewidth=2
    , marker = 'None'
    )

  ax.set_xlabel(xlabel)
  ax.set_ylabel(ylabel)

  return ax

# here is where the training starts.

tracklayers = [ 32 , 32 , 32 , 32 , 32 ]
jetlayers = [ 64 , 64 , 64 , 64 , 64 ]

def buildModel(tlayers, jlayers, ntargets):
  inputs = layers.Input(shape=(None, tlayers[0]))

  outputs = inputs
  outputs = layers.Masking(mask_value=MASKVAL)(outputs)

  for nodes in tlayers[:-1]:
    outputs = layers.TimeDistributed(layers.Dense(nodes, activation='relu'))(outputs)
    outputs = layers.BatchNormalization()(outputs)

  outputs = layers.TimeDistributed(layers.Dense(tlayers[-1], activation='softmax'))(outputs)
  outputs = Sum()(outputs)

  for nodes in jlayers:
    outputs = layers.Dense(nodes, activation='relu')(outputs)
    outputs = layers.BatchNormalization()(outputs)

  outputs = layers.Dense(ntargets + ntargets*(ntargets+1)//2)(outputs)

  return \
    keras.Model \
    ( inputs = inputs
    , outputs = outputs
    )


# TODO
# this ignores any dimension beyond the first!
def LogNormal1D(true, meanscovs):
  ntargets = true.shape[1]
  means = meanscovs[:,:ntargets]
  # ensure diagonal is positive
  logsigma = meanscovs[:,ntargets:2*ntargets]
  rest = meanscovs[:,2*ntargets:]

  # TODO
  # build matrix

  return ((means[:,0] - true[:,0])**2 / (2*keras.backend.exp(logsigma[:,0])**2)) + logsigma[:,0]

"""
model = buildModel([len(trackfeatures)] + tracklayers, jetlayers, 1)

model.summary()

model.compile \
  ( loss = LogNormal1D
  , optimizer = keras.optimizers.Adam(learning_rate=LR)
  )

for i in range(EPOCHS):
  print("epoch number:", i)
  model.fit(tracks, bhads, batch_size=BATCHSIZE)

  pred = model.predict(tracks[:10])
  
  print(pred[:,0] - bhads[:10])
  print(numpy.exp(pred[:,1]))
  print((pred[:,0] - bhads[:10]) / numpy.exp(pred[:,1]))

pred = model.predict(tracks)

numpy.save("/home/physics/phujdj/DeepLearningParticlePhysics/pred.npy",pred)
numpy.save("/home/physics/phujdj/DeepLearningParticlePhysics/true.npy",bhads)
"""