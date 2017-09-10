#include "MIL/ROISelector.hpp"

using namespace cv;

namespace MIL
{

Rect selectROI(InputArray img, bool showCrosshair, bool fromCenter)
{
    ROISelector selector;
    return selector.select("ROI selector", img.getMat(), showCrosshair, fromCenter);
}

Rect selectROI(const String& windowName, InputArray img, bool showCrosshair, bool fromCenter)
{
    ROISelector selector;
    return selector.select(windowName, img.getMat(), showCrosshair, fromCenter);
}

void selectROIs(const String& windowName, InputArray img,
                             std::vector<Rect>& boundingBox, bool showCrosshair, bool fromCenter)
{
    ROISelector selector;
    selector.select(windowName, img.getMat(), boundingBox, showCrosshair, fromCenter);
}


} /* namespace MIL */
