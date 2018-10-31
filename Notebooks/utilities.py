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
    note_freq = frequencies('C0', 'C8')
    rows = []
    for i in range(1, len(note_freq)-1):
        low  = note_freq[i] - (1/2)*(note_freq[i]-note_freq[i-1])
        high = note_freq[i] + (1/2)*(note_freq[i+1]-note_freq[i])
        ind  = (freq>=low)&(freq<=high)
        row  = spec[ind,:].sum(0) # Average it
        if (ind.sum() > 0):
             row /= np.mean(ind)
        rows.append(row)
    data = np.asarray(rows)
    return data, note_freq[1:-1]


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