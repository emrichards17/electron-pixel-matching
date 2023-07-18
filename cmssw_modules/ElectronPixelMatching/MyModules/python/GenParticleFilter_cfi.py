import FWCore.ParameterSet.Config as cms


genParticleFilter = cms.EDFilter(
    "GenParticleFilter",
    
    label_generator = cms.untracked.InputTag("generator"),
    label_genParticle = cms.untracked.InputTag("genParticles"),
    
    atLeastN = cms.int32(1),
    pdgIds = cms.vint32(0),
    
    minPt = cms.double(0.0),
    maxPt = cms.double(99999.0),
            
    minAbsEta = cms.double(0.0),
    maxAbsEta = cms.double(9999.0),
    
    isGunSample = cms.bool(False),
    
    debug = cms.bool(False),
)
