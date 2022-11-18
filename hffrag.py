import uproot
import awkward
import numpy
import matplotlib.figure as figure


# this opens up the root file and points to the "CharmAnalysis" tree inside.
tree = uproot.open("hffrag.root:CharmAnalysis")


# decide which branches of the tree we actually want to look at
# uncomment as needed!
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
  # , "AnalysisAntiKt4TruthJets_ghostB_pt"
  # , "AnalysisAntiKt4TruthJets_ghostB_eta"
  # , "AnalysisAntiKt4TruthJets_ghostB_phi"
  # , "AnalysisAntiKt4TruthJets_ghostB_m"
  

  # reconstructed jet information
  # , "AnalysisJets_pt_NOSYS"
  # , "AnalysisJets_eta"
  # , "AnalysisJets_phi"
  # , "AnalysisJets_m"


  # reconstructed track information
  # , "AnalysisTracks_pt"
  # , "AnalysisTracks_eta"
  # , "AnalysisTracks_phi"
  ]


# read in the requested branches from the file
features = tree.arrays(branches)


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
  fig = figure.Figure(figsize=(8, 8),facecolor= "white")
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


# this creates a histogram of the truth jet pTs.
truthjets = features["AnalysisAntiKt4TruthJets_pt"] 

fig = \
  binneddensity \
  ( truthjets[:,0] # the first jet in each event
  , fixedbinning(0, 200000, 50) # 50 bins from 0 to 200000 MeV
  , xlabel="first truth jet $p_T$ [MeV]"
  )
  
fig.savefig("truthjet-pt.png")

# this counts the number of b-hadron identification numbers and prints out the
# counts.
# these are defined in the PDG (but no need to worry about the details for now!)
# https://pdg.lbl.gov/2007/reviews/montecarlorpp.pdf


truthbhadronsid = \
  awkward.flatten \
  ( features["AnalysisAntiKt4TruthJets_ghostB_pdgId"]
  , axis=None
  )

# loop over all the b-hadrons associated to jets
counts = {}
for bhadid in truthbhadronsid:
  if bhadid in counts:
    counts[bhadid] += 1
  else:
    counts[bhadid] = 1

print()
for bhadid in counts:
  print("PDG ID:", bhadid)
  print("number of b-hadrons:", counts[bhadid])
  print()
