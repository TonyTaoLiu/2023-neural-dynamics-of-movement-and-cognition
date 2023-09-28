import os, sys, pathlib

GoodDataList = {'dualArea':{}, 'M1':{}, 'PMd':{}}

#------------------------------------

GoodDataList['dualArea']['Chewie'] = ['Chewie_CO_VR_2016-09-14.mat',
                                      'Chewie_CO_CS_2016-10-21.mat',
                                      'Chewie_CO_FF_2016-10-05.mat',
                                      'Chewie_CO_CS_2016-10-14.mat',
                                      'Chewie_CO_FF_2016-10-13.mat'
                                     ]

GoodDataList['dualArea']['Mihili'] = ['Mihili_CO_VR_2014-03-06.mat',
                                      'Mihili_CO_VR_2014-03-03.mat',
                                      'Mihili_CO_FF_2014-02-17.mat',
                                      'Mihili_CO_VR_2014-03-04.mat'
                                     ]

GoodDataList['dualArea']['MrT'] = ['MrT_CO_VR_2013-09-09.mat',
                                   'MrT_CO_VR_2013-09-05.mat'
                                  ]

#-----------------------------------

GoodDataList['M1']['Chewie'] = ['Chewie_CO_VR_2016-09-14.mat',
                                'Chewie_CO_FF_2016-10-13.mat',
                                'Chewie_CO_CS_2016-10-21.mat',
                                'Chewie_CO_CS_2016-10-14.mat'
                               ]

GoodDataList['M1']['Chewie2'] = ['Chewie_CO_CS_2015-03-19.mat',
                                 'Chewie_CO_CS_2015-03-13.mat',
                                 'Chewie_CO_CS_2015-03-11.mat',
                                 'Chewie_CO_CS_2015-03-12.mat'
                                ]

GoodDataList['M1']['Mihili'] = ['Mihili_CO_VR_2014-03-06.mat',
                                'Mihili_CO_VR_2014-03-03.mat',
                                'Mihili_CO_FF_2014-02-17.mat'
                               ]

GoodDataList['M1']['Jaco'] = ['Jaco_CO_CS_2016-01-28.mat',
                              'Jaco_CO_CS_2016-02-04.mat',
                              'Jaco_CO_CS_2016-02-17.mat'
                             ]

#-----------------------------------

GoodDataList['PMd']['Chewie'] = ['Chewie_CO_FF_2016-09-21.mat',
                                 'Chewie_CO_VR_2016-09-14.mat',
                                 'Chewie_CO_FF_2016-09-15.mat',
                                 'Chewie_CO_FF_2016-09-19.mat'
                                ]

GoodDataList['PMd']['Mihili'] = ['Mihili_CO_FF_2014-02-18.mat',
                                 'Mihili_CO_CS_2014-09-29.mat',
                                 'Mihili_CO_FF_2014-02-17.mat'
                                ]

GoodDataList['PMd']['MrT'] = ['MrT_CO_VR_2013-09-05.mat',
                              'MrT_CO_VR_2013-09-09.mat',
                              'MrT_CO_FF_2013-08-21.mat'
                             ]

MCx = {}
for area in ['M1', 'PMd', 'dualArea']:
    for animal in GoodDataList[area]:
        if animal not in MCx:
            MCx[animal] = []
        MCx[animal].extend(GoodDataList[area][animal])
        MCx[animal] = list(set(MCx[animal]))

GoodDataList['MCx'] = MCx