def note_position(a):
    notes = {'A':9, 'A#':10, 'B':11, 'C':0, 'C#':1, 'D':2, 
             'D#':3, 'E':4, 'F':5, 'F#':6, 'G':7, 'G#':8}
    note = notes[a[:-1]]
    scale = int(a[-1])
    return note + scale*12

def note_distance(a, b):
    return note_position(b)-note_position(a)

def frequencies(start, stop, A4=440):
    import numpy as np
    a = 2**(1/12)
    return a ** np.arange(note_distance('A4', start), note_distance('A4', stop)) * A4

def note_names(start, stop):
    notes = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
    
    note = notes.index(start[:-1])
    scale = int(start[-1])
    
    stop_note = notes.index(stop[:-1])
    stop_scale = int(stop[-1])
    
    output = []
    
    while scale < stop_scale or (scale == stop_scale and note < stop_note):
        output.append(notes[note] + str(scale))
        note += 1
        if note >= len(notes):
            note = 0
            scale += 1
        
    return output

def make_corrections(spec, freq):
    import numpy as np
    note_freq = frequencies('A0', 'C#8') # end is not included
    rows = []
    a = 2**(1/12)
    lo_factor = 0.5*(1+1/a)
    hi_factor = 0.5*(1+a) 
    df = freq[1] - freq[0]
    for i in range(len(note_freq)):
        lo = note_freq[i]*lo_factor
        hi = note_freq[i]*hi_factor
        ind = (freq>=lo)&(freq<=hi)
        
        #if lo <= 1174.66 <= hi:
#         print(np.where(ind), note_freq[i], lo, hi) #, repr(spec[ind,:]))
        
        if ind.sum() != 0:
            #row = spec[ind,:].max(0)
            row = normal_peak(spec[ind,:], df)
        else:
            row = np.zeros(spec.shape[1])
        rows.append(row)
    data = np.vstack(rows)
    return data, note_freq

def spect(wav, fs):
    import numpy as np
    smallest_nps = fs/24 * 4/3 
    smallest_nps = 2**int(np.log2(smallest_nps))
    note_freq = frequencies('A0', 'D8')
    df = np.diff(note_freq)
    nps = np.maximum(np.int64(2**np.ceil(np.log2(fs/df))), smallest_nps)
    nps_uniq = list(np.unique(nps))
    
    data = []
    for nperseg in nps_uniq:
        data.append(sg.spectrogram(wav, fs, nperseg=nperseg, noverlap=nperseg-smallest_nps))
    a = 2**(1/12)
    lo_factor = 0.5*(1+1/a)
    hi_factor = 0.5*(1+a) 
    
    full_spec = np.zeros((len(note_freq)-1, len(data[0][1])))
    for i,(nf,nperseg) in enumerate(zip(note_freq, nps)):
        nps_ind = nps_uniq.index(nperseg)
        freq, time, spec = data[nps_ind]
        lo = nf*lo_factor
        hi = nf*hi_factor
        ind = (freq>=lo)&(freq<=hi)
#         print(nf, ind.sum(), abs(freq[ind] - nf).min() / nf)
        df = freq[1] - freq[0]
        peak = normal_peak(spec[ind,:], df)
        full_spec[i,:len(time)] = peak

    return note_freq[:-1], data[0][1], full_spec


def display_ratios(corr, argsort, lbl='N/A'):
    import matplotlib.pyplot as plt
    plt.plot(corr.max(axis=0)[1:] / [corr[x,i] for i,x in enumerate(argsort[-2][1:], 1)], label=lbl)
    plt.xlabel('Time')
    plt.ylabel('Ratio (Inverted)')
    plt.legend()
    plt.gca().set_aspect('auto')
    
    
def display_spec(time, freq, spec, ylim=8192): 
    import matplotlib.pyplot as plt
    plt.ylim(0,ylim)
    plt.pcolormesh(time, freq, spec)
    plt.imshow(spec)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.gca().set_aspect('auto')
    plt.show()

def normal_peak(vals, dfreq):
    import numpy as np
    if vals.shape[0] == 1: return vals
    sums = vals.sum(0, keepdims=True) 
    weights = vals/sums # the weights for each time
    freqs = np.arange(1, vals.shape[0]+1)[:,None] # approximation of frequencies (exact values not needed)
    mean = (weights*freqs).sum(0, keepdims=True)
    var = (((freqs-mean)**2)*weights).sum(0)/dfreq
    peak = 1/np.sqrt(2*np.pi*var)
    peak *= sums.squeeze()
    return peak


def gather_data(filename):
    import numpy as np
    from scipy.io import wavfile
    import scipy.signal as sg
    fs, wav = wavfile.read(filename)
    wav = wav.astype(np.double)
    freq, time, spec = sg.spectrogram(wav, fs, nperseg=16384)
    arg_sort = np.argsort(spec,  axis=0)
    corrections, nf = make_corrections(spec,  freq)
    arg_sort_corr = np.argsort(corrections,  axis=0)
    return fs, wav, freq, time, spec, arg_sort, corrections, nf, arg_sort_corr
    
    
    
    
    