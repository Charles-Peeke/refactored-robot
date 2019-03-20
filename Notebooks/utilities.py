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
    import scipy.signal as sg
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
    freq, time, spectrogram = spect(wav, fs)
    arg_sort = np.argsort(spectrogram,  axis=0)
    return fs, wav, freq, time, spectrogram, arg_sort
    
def precompute_spect(fs=44100):
    """
    Precomputes things for TensorFlow spectrograms based on the frequency sampling rate.
    
    Returns:
        nps_uniq - a list of the unique n-per-segment that are needed
                   they are in sorted order so nps_uniq[0] gives the smallest nps
        dfreqs   - a list of the delta-frequencies about the spectrograms for each n-per-segment
                   in the same order as npw_uniq, each entry
        notes    - a list of information about each note, each entry is a tuple of:
                    * the index within nps_uniq and dfreqs to get other information
                    * the slice to use to extract the necessary frequencies from the spectrogram
    """
    import numpy as np
    
    # Calculate number per segment for each range
    smallest_nps = fs/24 * 4/3 
    smallest_nps = 2**int(np.log2(smallest_nps))
    note_freq = frequencies('A0', 'D8')
    df = np.diff(note_freq)
    nps = np.maximum(np.int64(2**np.ceil(np.log2(fs/df))), smallest_nps)
    nps_uniq = list(np.unique(nps))

    # Calculate the spectrogram frequencies
    freqs = []
    for nperseg in nps_uniq:
        freqs.append(np.fft.rfftfreq(nperseg, 1/fs))
    dfreqs = [freq[1]-freq[0] for freq in freqs]

    # Build information up about the notes
    a = 2**(1/12)
    lo_factor = 0.5*(1+1/a)
    hi_factor = 0.5*(1+a)
    notes = []
    for i,(nf,nperseg) in enumerate(zip(note_freq, nps)):
        nps_ind = nps_uniq.index(nperseg)
        freq  = freqs[nps_ind]
        start = np.argmax(freq>=(nf*lo_factor))
        end   = np.argmax(freq> (nf*hi_factor))
        #notes.append((nps_ind, slice(start, end, 1)))
        notes.append((nps_ind, (start, end)))

    return nps_uniq, dfreqs, notes
    
def gather_samples(notes=note_names('C1', 'C2'), nsamples=4):
    '''
    Gather all of the files used to train the model
    Trombone and Guitar to setup a binary network
    '''
    sample_files = []
    target_vals = []
    location = '../samples/'
    instruments = ['Trombone', 'Guitar', 'Piano']
    for i, name in enumerate(instruments):
        for note in notes:
            for j in range(1, nsamples+1):
                file_name = location +name+"/"+name +"_"+note[:-1]+"("+str(j)+").wav"
                sample_files.append(file_name)
                target_vals.append(i)
    return sample_files, target_vals

def tf_normal_peak(vals, dfreq):
    '''
    Calculate the normal peak of the frequencies with tensors
    '''
    import numpy as np
    import tensorflow as tf

    sums = tf.reduce_sum(vals, 0, keepdims=True)
    
    weights = tf.div_no_nan(vals,sums)
        
    freqs = tf.range(tf.cast(tf.shape(vals)[0], 'float64'), dtype='float64')
    
    mean = tf.reduce_sum(
        tf.cast(weights, 'float64') * tf.cast(freqs[:,None], 'float64'),  
        axis=0,
        keepdims=True)
    
    var = tf.truediv(tf.reduce_sum(tf.cast(((freqs[:,None]-mean)**2), 'float64')*tf.cast(weights, 'float64'), 0),dfreq)
    
    denom_var = tf.multiply(tf.constant(2.0, dtype='float64'), np.pi)
    
    denom_var_mult = tf.scalar_mul(denom_var, var)
    
    denom = tf.sqrt(tf.abs(denom_var_mult))
    
    dst_peak = tf.div_no_nan(tf.constant(1.0, dtype='float64'),denom)
#     with tf.Session() as sess:
#         print(sess.run(dst_peak))

    peak = (tf.cast(dst_peak, 'float64') * tf.cast(dfreq, 'float64') * tf.cast(tf.squeeze(sums), 'float64')) if (dst_peak != 'nan') else 0
    return peak


def preprocess(filename, nps_uniq, dfreqs, notes, smallest_nps, note_freq):
    from scipy.io import wavfile
    import numpy as np
    # Gathering the Data
    # Read the Wave File
    fs, wav = wavfile.read(filename)
    if wav.ndim == 2: wav = wav[:,0]
    wav = wav.astype(np.double)
    # Calculate the Spectrogram
#     for sec in range(8):
#         wav_sec = wav[sec*(len(wav)//8):(sec+1)*(len(wav)//8)]
#         print(len(wav_sec))
#     print(wav.shape)
    spectrogram = tf_spect(filename, wav, nps_uniq, dfreqs, notes, smallest_nps, note_freq)
#     print(spectrogram)
#     arg_sort = np.argsort(spectrogram,  axis=0)
    return spectrogram



def tf_spect(name, wav, nps_uniq, dfreqs, notes, smallest_nps, note_freq):  
    import tensorflow as tf
    import numpy as np
    data = []
    first = True
    # Caclculating the Spectrogram
    
    for nperseg in nps_uniq:
        signals = tf.convert_to_tensor(wav, tf.float32)
        stfts = tf.contrib.signal.stft(signals, frame_length=np.int32(3*nperseg - 2*smallest_nps),
                                       frame_step=np.int32(nperseg),
                                       fft_length=np.int32(3*nperseg - 2*smallest_nps))
        magnitude_spectrograms = tf.abs(stfts)
        data.append(magnitude_spectrograms)
    fs_raw = tf.Variable(tf.zeros((len(note_freq)-1, tf.shape(data[0])[0]), tf.float64), name='Raw_Freqs')
    assignments = []
    # Calculate the normal peaks in order to make a more clear spectrogram
    for i, (nps_ind, slc) in enumerate(notes):
        tf_spec = data[nps_ind]
        spec_slc = tf_spec[:,slc[0]:slc[1]]
        tf_np = tf.cast(spec_slc[:,0], tf.float64) if slc[1]-slc[0] == 1 else tf_normal_peak(tf.transpose(spec_slc), dfreqs[nps_ind])
        time = tf.shape(tf_spec)[0]
        assignments.append(fs_raw[i,:time].assign(tf_np))
        
    with tf.control_dependencies(assignments):
        full_spec = fs_raw.read_value()
    # Return the note frequencies, original data, and spectrogram with normalized peaks
    return full_spec#, fs_raw, assignments
    