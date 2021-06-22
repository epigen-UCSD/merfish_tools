import numpy as np
import re


class DaxFile:
    def __init__(self, filename, num_channels):
        self.filename = filename
        self.num_channels = num_channels
        self._info = None
        self._memmap = np.memmap(
            filename, dtype=np.uint16, mode='r',
            shape=(self.fileinfo('frames'), self.fileinfo('height'), self.fileinfo('width')))
        if self.fileinfo('endian') == 1:
            self._memmap = self._memmap.byteswap()

    def logmsg(self, message):
        return f"Dax file {self.filename} - {message}"

    def fileinfo(self, tag):
        if self._info is None:
            self.load_fileinfo()
        return self._info[tag]

    def load_fileinfo(self):
        with open(self.filename.replace('.dax', '.inf')) as f:
            infodata = f.read()
        self._info = {}
        m = re.search(r'frame dimensions = ([\d]+) x ([\d]+)', infodata)
        self._info['height'] = int(m.group(1))
        self._info['width'] = int(m.group(2))
        m = re.search(r'number of frames = ([\d]+)', infodata)
        self._info['frames'] = int(m.group(1))
        m = re.search(r' (big|little) endian', infodata)
        self._info['endian'] = 1 if m.group(1) == "big" else 0

    def channel(self, channel):
        return self._memmap[channel::self.num_channels, :, :]

    def get_entire_image(self):
        return self._memmap[:, :, :]

    def zslice(self, zslice, channel=None):
        """If channel is None, return all channels."""
        if channel is None:
            return self._memmap[zslice*self.num_channels:zslice*self.num_channels+self.num_channels, :, :]
        else:
            return self._memmap[channel+zslice*self.num_channels, :, :]

    def frame(self, frame):
        return self._memmap[frame, :, :]

    def max_projection(self, channel=None):
        if channel is None:
            return np.max(self._memmap[:, :, :], axis=0)
        else:
            return np.max(self._memmap[channel::self.num_channels, :, :], axis=0)

    def block_range(self, xr=None, yr=None, zr=None, channel=None):
        if xr is None:
            xr = (0, self.fileinfo('width'))
        if yr is None:
            yr = (0, self.fileinfo('height'))
        if zr is None:
            zr = (0, self.fileinfo('frames'))
        if xr[0] < 0:
            xr = (0, xr[1])
        if yr[0] < 0:
            yr = (0, yr[1])
        if zr[0] < 0:
            zr = (0, zr[1])
        if channel is None:
            zslice = slice(self.num_channels*zr[0], self.num_channels*(zr[1] + 1))
        else:
            zslice = slice(self.num_channels*zr[0] + channel, self.num_channels*(zr[1] + 1) + channel, self.num_channels)
        yslice = slice(yr[0], yr[1] + 1)
        xslice = slice(xr[0], xr[1] + 1)
        return self._memmap[zslice, yslice, xslice], (zslice, yslice, xslice)

    def block(self, center, volume, channel=None):
        zr = (center[0] - volume[0], center[0] + volume[0])
        yr = (center[1] - volume[1], center[1] + volume[1])
        xr = (center[2] - volume[2], center[2] + volume[2])
        return self.block_range(zr=zr, yr=yr, xr=xr, channel=channel)
