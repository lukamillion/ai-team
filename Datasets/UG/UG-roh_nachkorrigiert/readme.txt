readme.txt
Users/aw/UNI/BMBF/Auswertung/UG_roh/

cutting.f90 was developed in several steps to prepare the raw data. 


first step was to cut off frames, which were not part of the certain experiments.
by comparison with video data the special startframes and 
endframes were identified and saved in a special file (frames.txt). 
framenumbers were checked and frames outside of FieldOfInterest were dismissed.
file: cutoff.f90 


second step was to recount the person numbers. due to the cutoff there were breaks 
and wrong order of ped_nr so that they had to be re-numbered and re-ordered.
Raw data was checked again due to wrong order of data because of 
manually made correction of petrack-files: 
high person numbers with low frame numbers at the end of files should be located in beginning of file, 
because of designated starting frame "0".
file: recount.f90


third step was the re-numbering of frame_nr, so that all experiments were 
set to startingframe "0".
file: reframe.f90


All Fortran files were used with the XCode-Project: margin_cut.xcodeproj 
	and adjusted to particular use-cases.
