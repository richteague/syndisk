from astropy.convolution import Gaussian2DKernel
from astropy.convolution import convolve_fft
from astropy.io import fits
from gofish import imagecube
import numpy as np


class simulationcube(imagecube):
    """
    Read in a radiative transfer model and make quick simulated observations.

    Args:
        path (str): Relative path to the FITS cube.
        FOV (Optional[float]): Clip the image cube down to a specific
            field-of-view spanning a range ``FOV``, where ``FOV`` is in
            [arcsec].
        v_range (Optional[tuple]): A tuple of minimum and maximum velocities
            to clip the velocity range to.
        verbose (Optional[bool]): Whether to print out warning messages.
        primary_beam (Optional[str]): Path to the primary beam as a FITS file
            to apply the correction.
        bunit (Optional[str]): If no `bunit` header keyword is found, use this
            value, e.g., 'Jy/beam'.
        pixel_scale (Optional[float]): If no axis information is found in the
            header, use this value for the pixel scaling in [arcsec], assuming
            an image centered on 0.0".
    """

    def __init__(self, path, **kwargs):
        super().__init__(path=path, **kwargs)

    def synthetic_ALMA(self, bmaj=None, bmin=None, bpa=0.0, rms=None,
                       chan=None, nchan=None, vcent=None, dpix=None, npix=None,
                       spectral_response=None, filename=None, overwrite=False):
        """
        Generate synthetic ALMA observations by convolving the data spatially
        and spectrally and adding correlated noise. Will automatically convert
        the attached brightness unit to 'Jy/beam'. This will not capture any
        effects associated with spatial filtering or imaging real
        interferometric data.

        Args:
            bmaj (optional[float]): Beam major axis in [arcsec].
            bmin (optional[float]): Beam minor axis in [arcsec].
            bpa (optional[float]): Beam position angle in [degrees].
            rms (optional[float]): RMS noise of the noise.
            chan (optional[float]): Channel size (m/s) of the resulting data.
            nchan (optional[int]): Number of channels of the resulting data. If
                this would extend beyond the attached velocity then these edge
                channels are ignored.
            rescale (optional[int]): Rescaling factor for the pixels. If
                ``rescale='auto'`` then the pixels will be rescaled so there's
                5 pixels per bmin.
            spectral_response (optional[str]): Type of spectral response to
                include. ``'hanning'`` will include a triangle kernel, while
                ``'averageX'``, where ``'X'`` is a number will use a simple
                running average of ``X`` channels.
            save (optional[bool/str]): If True, save the data as a new cube.
                You may also provide a path to save to (noting that this will
                overwrite anything).

        Returns:
            ndarrays: If not saved to disk, returns the spatial axis, velocity
                axis and the datacube.
        """

        # Make beam kernel. Also sign the new pixel scaling to 1/5th of the
        # beam minor axis if not specified as per CASA docs recommendation.

        if bmaj is not None:
            bmin = bmaj if bmin is None else bmin
            dpix = bmin / 5.0 if dpix is None else dpix
            self.dpix_sc = dpix #rescaled pixel size, needed for _beamkernel
            beam = self._beamkernel(bmaj, bmin, bpa)
            if self.verbose and dpix * 5.0 > bmaj:
                print("WARNING: Specified `dpix` does not sample bmin well.")
        else:
            beam = False
            dpix = self.dpix if dpix is None else dpix

        # Rescale the data to the new dpix scale and cut down to the requested
        # number of pixels. If this is is larger than the rescaled image, raise
        # a warning and continue.

        #dpix = self.dpix if dpix is None else dpix #no longer needed
        npix = self.nxpix if npix is None else npix
        data = self._rescaled_data_spatial(dpix=dpix, npix=npix)
        npix = data.shape[-1]
        if self.verbose and (dpix != self.dpix):
            print('Rescaled image has a pixel scale of {:.3g}" '.format(dpix)
                  + 'and has a FOV of {:.2g}" '.format((npix - 1) * dpix)
                  + '({} x {} pixels).'.format(data.shape[-2], data.shape[-1]))
        axis = np.arange(npix) * dpix
        axis -= np.mean(axis)

        # Define the new velocity axis. Should be centered about the same
        # central velocity as the input data.

        chan = self.chan if chan is None else chan
        if nchan is None:
            nchan = int(np.floor((self.velax.max() - self.velax.min()) / chan))
        vcent = np.mean(self.velax) if vcent is None else vcent
        velax = self._new_velax(chan=chan, nchan=nchan, vcent=vcent)
        data = self._rescaled_data_velocity(data, velax)
        if self.verbose:
            print('Velocity axis spans {:.2g} km/s to '.format(velax[0] / 1e3)
                  + '{:.2g} km/s with '.format(velax[-1] / 1e3)
                  + '{} channels spaced by '.format(velax.size)
                  + '{:.0f} m/s.'.format(chan))

        # Convolve the data. This uses the astropy.convolution.convolve_fft
        # function which appears to be the fastest. Note that the accuracy has
        # not been tested. If necessary, convert from Jy/pixel to Jy/beam.
        # Check that the dpix should be the INPUT dpix, not the rescaled dpix.

        if beam:
            if self.verbose:
                print("Will convolve attached data with a Gaussian beam of "
                      + '{:.2f}" x {:.2f}" '.format(bmaj, bmin)
                      + '({:.1f} deg).'.format(bpa))
            data = [convolve_fft(d, beam, boundary='wrap') for d in data]
            data = np.squeeze(data)
            if 'jy/pix' in self.header['bunit'].lower():
                data *= np.pi * bmin * bmaj / 4. / np.log(2.) / self.dpix**2
                converted = True
            else:
                converted = False

        # Add the correlated noise.

        if rms is not None:
            if self.verbose:
                print("Adding correlated noise with an RMS "
                      + "of {:.2g} mJy/beam.".format(rms * 1e3))
            data = self._add_correlated_noise(data, rms, beam)

        # Include spectral response function.

        if spectral_response is not None:
            if self.verbose:
                print("Adding influence of spectral response function.")
            data = self._add_spectral_response(data, spectral_response)

        # Save the cube to a new FITS file.

        self._save_synthetic_ALMA(data, axis, velax, rms, converted, bmaj,
                                  bmin, bpa, filename, overwrite)

    def _beamkernel(self, bmaj, bmin=None, bpa=0.0):
        """
        Returns the 2D Gaussian kernel for convolution.

        Args:
            bmaj (float): Beam major FWHM in [arcsec].
            bmin (optional[float]): Beam minor FWHM in [arcsec]. Defaults to
                ``bmaj`` for a circular beam.
            bpa (optionaln[float]): Beam position angle in [deg]. Measured as
                the angle between the major FWHM and north in an anticlockwise
                direction. Defaults to 0 degrees.
        """
        bmin = bmaj if bmin is None else bmin
        bmaj = bmaj / self.dpix_sc / 2.0 / np.sqrt(2.0 * np.log(2.0))
        bmin = bmin / self.dpix_sc / 2.0 / np.sqrt(2.0 * np.log(2.0))
        return Gaussian2DKernel(bmin, bmaj, np.radians(90+bpa)) #bpa corrected to N axis

    def _new_velax(self, chan, nchan, vcent):
        """
        Defines the new velocity axis into which to interpolate the model.

        Args:
            chan (float): Channel spacing in [m/s].
            nchan (int): Number of channels.
            vcent (float): Centeral velocity of the new velocity axis.

        Returns:
            velax (array): New velocity axis in [m/s].
        """
        if self.verbose and (chan < self.chan):
            print("WARNING: New channel spacing ({:.0f} m/s) ".format(chan)
                  + "finer than input data ({:.0f} m/s).".format(self.chan))
        nchan = self.velax.size if nchan is None else nchan
        velax = (nchan - 1) * chan * np.linspace(-0.5, 0.5, nchan) + vcent
        velax = velax[np.logical_and(velax >= self.velax.min(),
                                     velax <= self.velax.max())]
        return velax.astype('float')

    def _rescaled_data_spatial(self, dpix, npix):
        """
        Rescale the data in the spatial dimension using ``scipy.ndimage.zoom``.

        Args:
            dpix (float): Pixel scale in [arcsec] of the rescaled data.
            npix (int): Number of pixels on each side of the channel.

        Returns:
            data (array): Array of channels of shape (npix, npix).
        """
        data = np.copy(self.data)
        if dpix / self.dpix < 1:
            raise ValueError('Requested `dpix` ({:.3f}") '.format(dpix)
                             + 'smaller than attached '
                             + '`dpix` ({:.3f}").'.format(self.dpix))
        elif dpix / self.dpix == 1:
            return data
        else:
            from scipy.ndimage import zoom
            data = [zoom(d, self.dpix / dpix) for d in data]
            data = np.squeeze(data)

        npix = data.shape[-1] if npix is None else npix
        if npix > data.shape[-1]:
            if npix != self.data.shape[-1]:
                print("Requested `npix` larger than input image;"
                      + " ignoring `npix`.")
            npix = data.shape[-1]

        spix = int((data.shape[-1] - npix) / 2)
        fpix = spix + npix + 1
        if data.ndim == 3:
            data = data[:, spix:fpix, spix:fpix]
        else:
            data = data[spix:fpix, spix:fpix]
        assert data.shape[-1] == npix, "Wrong number of pixels."
        assert data.shape[-2] == npix, "Wrong number of pixels."
        return data

    def _rescaled_data_velocity(self, data, velax):
        """
        Resample the data onto a new velocity axix with a cubic spline
        interpolation.

        NOTE: Current implementation is just to take the interpolated value
        between the current channels (i.e. not integrating between channels as
        suggested by the RADMC-3D manual). Reference Andrews et al. (in prep.)

        Args:
            data (array): Data to resample on a new velocity axis.
            velax (array): New velocity axis onto which to sample the data in
                [m/s].

        Returns:
            data_interp (array): Resampled data.
        """
        from scipy.interpolate import CubicSpline
        assert data.shape[0] == self.velax.size, "Mismatch in data velocity."
        data_interp = CubicSpline(self.velax, data, axis=0)(velax)
        assert data_interp.shape[0] == velax.size, "Mismatch in data velocity."
        return data_interp

    def _add_correlated_noise(self, data, rms, beam=False):
        """
        Return correlated noise with the same shape as provided data.

        Args:
            data (array): Data to add correlated noise to.
            rms (float): RMS of the noise to add.
            beam (kernel): Astropy convolution kernel to convolve the noise.

        Returns:
            data (array): Data with the correlated noise added.
        """
        noise = np.random.normal(size=data.size).reshape(data.shape)
        if beam and data.ndim == 3:
            noise = [convolve_fft(n, beam, boundary='wrap') for n in noise]
        elif beam:
            noise = convolve_fft(noise, beam, boundary='wrap')
        return data + np.squeeze(noise) * rms / np.std(noise)

    def _add_spectral_response(self, data, spectral_response):
        """
        Include a spectral response function.

        Args:
            data (array): Data to include spectral response to.
            spectral_response (str,list): Either the name of the spectral
                response function to use, or a convolution kernel.

        Returns:
            data (array): Data with the spectral response function applied.
        """
        from scipy.ndimage import convolve1d
        if spectral_response == 'hanning':
            kernel = np.array([0.25, 0.5, 0.25])
        elif spectral_response == 'boxcar':
            kernel = np.array([0.5, 0.5])
        elif type(kernel) is str:
            raise ValueError("Unknown spectralc response "
                             + " function {}.".format(spectral_response))
        kernel = np.squeeze(kernel) / np.sum(kernel)
        return convolve1d(data, kernel, mode='wrap', axis=0)

    def _save_synthetic_ALMA(self, data, axis, velax, rms, converted,
                             bmaj=None, bmin=None, bpa=None, filename=None,
                             overwrite=False):
        """
        Save the synethic ALMA observation as a FITS file.

        Args:
            data (array): Data to save.
            axis (array): Spatial axis of the data.
            velax (array): Velocity axis of the data.
            converted (bool): Whether Jy/pix units were transformed to Jy/beam.
            bmaj (optional[float]): Beam major FWHM in [arcsec].
            bmin (optional[float]): Beam minor FWHM in [arcsec].
            bpa (optional[float]): Beam position angle in [deg].
            filename (optional[str]): Filename to save the data to.
            overwrite (optional[bool]): Whether to overwrite FITS files.

        """

        # Check the data is how we expect it.

        assert data.shape[0] == velax.size, "Wrong number of channels in data."
        assert data.shape[1] == axis.size, "Wrong number of y pixels in data."
        assert data.shape[2] == axis.size, "Wrong number of x pixels in data."
        dpix = np.diff(axis).mean()
        chan = np.diff(velax).mean()

        # Open and attach data.

        hdu = fits.PrimaryHDU()
        hdu.data = data

        # Right-Ascension axis.

        hdu.header['CTYPE1'] = 'RA---SIN'
        hdu.header['CDELT1'] = -dpix / 3600.
        hdu.header['CRPIX1'] = data.shape[1] / 2 + 1
        hdu.header['CRVAL1'] = axis[0] / 3600.
        hdu.header['CUNIT1'] = 'deg'

        # Declination axis.

        hdu.header['CTYPE2'] = 'DEC--SIN'
        hdu.header['CDELT2'] = dpix / 3600.
        hdu.header['CRPIX2'] = data.shape[1] / 2 + 1
        hdu.header['CRVAL2'] = axis[0] / 3600.
        hdu.header['CUNIT2'] = 'deg'

        # Velocity axis.

        hdu.header['CTYPE3'] = 'VELO-LSR'
        hdu.header['CDELT3'] = chan
        hdu.header['CRPIX3'] = 1
        hdu.header['CRVAL3'] = velax[0]
        hdu.header['CUNIT3'] = 'm/s'

        # Brightness unit.

        if converted:
            hdu.header['BUNIT'] = 'JY/BEAM'
        else:
            try:
                hdu.header['BUNIT'] = self.header['BUNIT']
            except ValueError:
                print("WARNING: No `BUNIT` value found. Assuming Jy/pix." +
                      "Can use `bunit` argument when reading in cube.")

        # Rest frequency. If none found, defaults to CO (3-2).

        try:
            hdu.header['RESTFREQ'] = self.header['RESTFREQ']
        except KeyError:
            print("WARNING: No `RESTFREQ` header value found."
                  + " Assuming CO (3-2) rest frequency.")
            hdu.header['RESTFREQ'] = 345.7959899e9

        # Beam properties.

        if bmaj is not None:
            hdu.header['BMAJ'] = bmaj / 3600.
            hdu.header['BMIN'] = bmin / 3600.
            hdu.header['BPA'] = bpa

        # Filename if none is given.

        if filename is None:
            if bmaj is not None:
                filename = '.{:.5g}arcsec'.format(np.mean([bmaj, bmin]))
            else:
                filename = ''
            filename += '.{:.3g}ms'.format(chan)
            if rms is not None:
                filename += '.{:.3g}mJybeam'.format(rms * 1e3)
            filename = self.path.replace('.fits', filename + '.fits')
        filename = filename.replace('.fits', '') + '.fits'
        if self.verbose:
            print("Saving cube to '{}'.".format(filename.split('/')[-1]))

        hdu.writeto(filename, overwrite=overwrite, output_verify='fix')
