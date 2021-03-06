import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.spines import Spine
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from sklearn import preprocessing

def radar_factory(num_vars, frame='circle'):
    """Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle' | 'polygon'}
        Shape of frame surrounding axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)
    # rotate theta such that the first axis is at the top
    theta += np.pi/2

    def draw_poly_patch(self):
        verts = unit_poly_verts(theta)
        return plt.Polygon(verts, closed=True, edgecolor='k')

    def draw_circle_patch(self):
        # unit circle centered on (0.5, 0.5)
        return plt.Circle((0.5, 0.5), 0.5)

    patch_dict = {'polygon': draw_poly_patch, 'circle': draw_circle_patch}
    if frame not in patch_dict:
        raise ValueError('unknown value for `frame`: %s' % frame)

    class RadarAxes(PolarAxes):

        name = 'radar'
        # use 1 line segment to connect specified points
        RESOLUTION = 1
        # define draw_frame method
        draw_patch = patch_dict[frame]

        def fill(self, *args, **kwargs):
            """Override fill so that line is closed by default"""
            closed = kwargs.pop('closed', True)
            return super(RadarAxes, self).fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super(RadarAxes, self).plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.concatenate((x, [x[0]]))
                y = np.concatenate((y, [y[0]]))
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            return self.draw_patch()

        def _gen_axes_spines(self):
            if frame == 'circle':
                return PolarAxes._gen_axes_spines(self)
            # The following is a hack to get the spines (i.e. the axes frame)
            # to draw correctly for a polygon frame.

            # spine_type must be 'left', 'right', 'top', 'bottom', or `circle`.
            spine_type = 'circle'
            verts = unit_poly_verts(theta)
            # close off polygon by repeating first vertex
            verts.append(verts[0])
            path = Path(verts)

            spine = Spine(self, spine_type, path)
            spine.set_transform(self.transAxes)
            return {'polar': spine}

    register_projection(RadarAxes)
    return theta

def unit_poly_verts(theta):
    """Return vertices of polygon for subplot axes.

    This polygon is circumscribed by a unit circle centered at (0.5, 0.5)
    """
    x0, y0, r = [0.5] * 3
    verts = [(r*np.cos(t) + x0, r*np.sin(t) + y0) for t in theta]
    return verts

def load_data(in_file):
    input_f = open(in_file, "r")
    matrix = []
    for line in input_f:
        channels = [] 
        for l in line.split("|"):
            samples = l.split(";")
            channels.append([float(i) for i in samples])
        matrix.append(channels)        
    input_f.close()
    return matrix

def load_labels(in_file):
    input_f = open(in_file, "r")
    labels = []
    for line in input_f:
        if ";" in line:
            labels.append(line.replace("\n","").split(";"))
        else:
            labels.append(line.replace("\n",""))
    input_f.close()
    return labels    

def get_data():
    AXES = ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8"] 
    X = load_data("e-nose_data/data/data_new.txt")
    X_max = []
    S_max = []
    for x in X:
        s_m = []
        for s in x:
            s_m.append(max(np.array(s)))
        S_max.append(s_m)
    for x,s_max in zip(X,S_max):        
        #print ("s_max: ", s_max)
        m = list(map(list, zip(*x))) 
        X_m = []
        for s in m:
            s_new = []
            for i,ss in zip(s,s_max):
                if i != ss:
                    s_new.append(0)
                else:
                    s_new.append(i)
            X_m.append(s_new)
        X_max.append(X_m)
            
    Y = load_labels("e-nose_data/data/labels_new.txt")
    data = [] 
    for x,y in zip(X_max,Y):
        data.append([AXES, tuple((y, x))])
    return data

if __name__ == '__main__':
    AXES = ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8"] 
    #X_train = load_data("e-nose_data/data/data_train.txt")
    N = len(AXES)
    data = get_data()
    print (len(data))
    theta = radar_factory(N, frame='polygon')
    
    for i in range(0,len(data)):
        data_block = data[i]
        spoke_labels = data_block.pop(0)

        plt.clf()
        plt.cla()

        fig = plt.figure(figsize=(9, 9))
        fig.subplots_adjust(wspace=0.25, hspace=0.20, top=0.85, bottom=0.05)

        colors = ['seagreen'] * 121
        # Plot the four cases from the example data on separate axes
        for n, (title, case_data) in enumerate(data_block):
            ax = fig.add_subplot(1, 1, 1, projection='radar')
            plt.rgrids()
            ax.set_title("Object "+str(i+1), weight='bold', size='medium', position=(0.5, 1.1),
                         horizontalalignment='center', verticalalignment='center')
            for d, color in zip(case_data, colors):
                ax.plot(theta, d, color=color)
                ax.fill(theta, d, facecolor=color, alpha=0.25)
            ax.set_varlabels(spoke_labels)

        # add legend relative to top-left plot
        # plt.subplot(1, 1, 1)
        # labels = ("S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8")
        # legend = plt.legend(labels, loc=(0.9, .95), labelspacing=0.1)
        # plt.setp(legend.get_texts(), fontsize='small')

        plt.figtext(0.5, 0.965, 'Volatile organic compound radar plot',
                    ha='center', color='black', weight='bold', size='large')
        #plt.show()
        #print ("saving e-nose_data/graphs/radar/test/"+"_".join(data_block[0][0])+"_"+str(i)+".png")
        #plt.savefig("e-nose_data/graphs/radar/test/"+"_".join(data_block[0][0])+"_"+str(i)+".png", dpi=100)
        print ("saving e-nose_data/graphs/radar/new/"+str(i+1)+"_"+data_block[0][0].replace(" ", "_")+".png")
        plt.savefig("e-nose_data/graphs/radar/new/"+str(i+1)+"_"+data_block[0][0].replace(" ", "_")+".png", dpi=100)
        plt.close('all')