using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SignTextReco
{
    class SignReco
    {
        /////////////////////////////////////////////第三版：轮廓寻找ROI////////////////////////////////////////////////
        public List<System.Drawing.Point> GetPlusSignPostion(Bitmap image)
        {
            var resList = new List<System.Drawing.Point>();
            using (Mat srcImg = OpenCvSharp.Extensions.BitmapConverter.ToMat(image))
            {
                Mat grayImg;
                try
                {
                    grayImg = srcImg.CvtColor(ColorConversionCodes.BGR2GRAY);
                }
                catch (OpenCvSharp.OpenCVException)
                {
                    grayImg = srcImg;
                }
                var binaryImg = new Mat();
                Cv2.Threshold(grayImg, binaryImg, 200, 255, ThresholdTypes.BinaryInv); //反向二值化

                /////////尝试计算所有外接轮廓,计算轮廓面积和中心点的像素值去冗余
                var rectList = ConditionNMS(binaryImg);
                foreach (var i in rectList)
                {
                    var centerPoint = new System.Drawing.Point(i.X + i.Width / 2, i.Y + i.Height / 2);
                    resList.Add(centerPoint);
                }
            }
            return resList;
        }
        private List<Rect> ConditionNMS(Mat binaryImg)
        {
            var rectList = new List<Rect>();
            var indexer = binaryImg.GetGenericIndexer<Vec3b>();

            var contours = new OpenCvSharp.Point[][] { };
            var hierarchy = new OpenCvSharp.HierarchyIndex[] { };
            var contourAreaList = new List<double>() { };
            var contourCenterList = new List<System.Drawing.Point>() { };
            Cv2.FindContours(binaryImg, out contours, out hierarchy, RetrievalModes.External, ContourApproximationModes.ApproxNone);
            for (int i = 0; i < hierarchy.Length; i++)
            {
                //contourAreaList.Add(Cv2.ContourArea(contours[i]));
                //精炼出的输入为binary、一个加号还是箭头的flag，返回的是有效的rect类型的List，加号需要 rect的中心点；箭头需要rect的两边坐标以及方向
                //var S = Cv2.ContourArea(contours[i]);
                var M = Cv2.Moments(contours[i]);
                if (M.M00 != 0)
                {
                    var cx = (int)(M.M10 / M.M00);
                    var cy = (int)(M.M01 / M.M00);
                    if (indexer[cy, cx].Item0 == 255)//相同条件1:反向二值图的轮廓中心为0
                                                     //相同条件2：四角为0
                    {
                        var rect = Cv2.BoundingRect(contours[i]);   //相同条件3：截出来的框轮廓数量为4

                        try
                        {
                            if (indexer[rect.Y, rect.X].Item0 == 0 && indexer[rect.Y + rect.Height, rect.X + rect.Width].Item0 == 0
                                && indexer[rect.Y + rect.Height, rect.X].Item0 == 0 && indexer[rect.Y, rect.X + rect.Width].Item0 == 0)
                            {
                                //Rect r = new Rect(new OpenCvSharp.Point(rect.X, rect.Y), new OpenCvSharp.Size(rect.Width, rect.Height));
                                //Cv2.Rectangle(binaryImg, r, Scalar.LimeGreen, 1);
                                //Cv2.DrawContours(srcImg, contours, i, Scalar.Gray, 2, LineTypes.Link8, hierarchy, 4);
                                var tempROI = new Mat();
                                binaryImg[rect].CopyTo(tempROI);//截出连通域

                                if (Math.Max(rect.Width, rect.Height) / Math.Min(rect.Width, rect.Height) < 1.2 && rect.Width < 200
                                    && PlusCondition(tempROI))  //plus
                                    rectList.Add(rect);
                            }
                        }
                        catch (Exception er)
                        {
                            //do nothing.
                        }
                    }
                }
            }
            return rectList;
        }
        private bool PlusCondition(Mat tempROI)
        {
            bool flag = false;
            Cv2.Threshold(tempROI, tempROI, 200, 255, ThresholdTypes.BinaryInv); //反向二值化

            var tempContours = new OpenCvSharp.Point[][] { };
            var tempHierarchy = new OpenCvSharp.HierarchyIndex[] { };
            Cv2.FindContours(tempROI, out tempContours, out tempHierarchy, RetrievalModes.Tree, ContourApproximationModes.ApproxNone);

            //对于每一个tempROI，对于箭头 或者 判断是否符合条件
            if (tempContours.GetLength(0) == 4)
            {
                var pointListX = new List<double>();//4个轮廓中心点
                var pointListY = new List<double>();
                var euDisList = new List<double>();
                double sumX = 0, sumY = 0;

                for (int i = 0; i < 4; i++)
                {
                    var M = Cv2.Moments(tempContours[i]);
                    if (M.M00 != 0)
                    {
                        var cx = (M.M10 / M.M00);
                        var cy = (M.M01 / M.M00);
                        pointListX.Add(cx);
                        pointListY.Add(cy);
                        sumX += cx;
                        sumY += cy;
                    }
                    else
                        return false;

                }
                var centerPointX = sumX / 4;
                var centerPointY = sumY / 4;
                for (int i = 0; i < 4; i++)
                {
                    euDisList.Add(Math.Pow(Math.Pow(Math.Abs(centerPointX - pointListX[i]), 2) + Math.Pow(Math.Abs(centerPointY - pointListY[i]), 2), 0.5));
                }
                var stdDev = CalculateStdDev(euDisList);

                if (stdDev < 0.12)
                {
                    flag = true;
                    // Console.WriteLine(stdDev);
                }
            }
            return flag;
        }
        private static double CalculateStdDev(List<double> values)
        {
            double ret = 0;

            //  计算平均数   
            double avg = values.Average();
            //  计算各数值与平均数的差值的平方，然后求和 
            double sum = values.Sum(d => Math.Pow(d - avg, 2));
            //  除以数量，然后开方
            ret = Math.Sqrt(sum / values.Count());
            return ret;
        }


        private double CosineCompare(List<double> a, List<double> b)
        {
            int listLen = Math.Min(a.Count, b.Count);
            a.Sort((n, m) => m.CompareTo(n));
            b.Sort((n, m) => m.CompareTo(n));
            double x = 0, y = 0, z = 0;
            double cosineRes;
            for (int i = 0; i < listLen; i++)
            {
                x += a[i] * a[i];
                y += b[i] * b[i];
                z += a[i] * b[i];
            }
            x = Math.Sqrt(x);
            y = Math.Sqrt(y);
            cosineRes = z / (x * y);

            return cosineRes;
        }
        public List<KeyValuePair<System.Drawing.Point[], int>> GetArrowsSignPostion(Bitmap image)
        {
            var resList = new List<KeyValuePair<System.Drawing.Point[], int>>();
            using (Mat srcImg = OpenCvSharp.Extensions.BitmapConverter.ToMat(image))
            {

                Mat grayImg;
                try
                {
                    grayImg = srcImg.CvtColor(ColorConversionCodes.BGR2GRAY);
                }
                catch (OpenCvSharp.OpenCVException)
                {
                    grayImg = srcImg;
                }
                var binaryImg = new Mat();
                Cv2.Threshold(grayImg, binaryImg, 200, 255, ThresholdTypes.BinaryInv);
                var orientationList = new List<int>();
                var rectList = ConditionNMS(binaryImg, ref orientationList);

                for (int i = 0; i < rectList.Count; i++)
                {
                    var beginPoint = new System.Drawing.Point();
                    var endPoint = new System.Drawing.Point();

                    //判断箭头方向
                    var pixelROIMat = new Mat();
                    binaryImg[rectList[i]].CopyTo(pixelROIMat);//截出连通域                    
                    var ROIIndexer = pixelROIMat.GetGenericIndexer<Vec3b>();

                    int arrowType = 0;
                    if (orientationList[i] == 1)  //正方向箭头
                    {
                        beginPoint = new System.Drawing.Point(rectList[i].X, rectList[i].Y + rectList[i].Height / 2);
                        endPoint = new System.Drawing.Point(rectList[i].X + rectList[i].Width, rectList[i].Y + rectList[i].Height / 2);
                    }
                    else    //反方向
                    {
                        endPoint = new System.Drawing.Point(rectList[i].X, rectList[i].Y + rectList[i].Height / 2);
                        beginPoint = new System.Drawing.Point(rectList[i].X + rectList[i].Width, rectList[i].Y + rectList[i].Height / 2);
                    }
                    resList.Add(new KeyValuePair<System.Drawing.Point[], int>(new System.Drawing.Point[] { beginPoint, endPoint }, arrowType));
                }
            }
            return resList;
        }
        private List<Rect> ConditionNMS(Mat binaryImg, ref List<int> orientationList)
        {
            var rectList = new List<Rect>();
            var indexer = binaryImg.GetGenericIndexer<Vec3b>();

            var contours = new OpenCvSharp.Point[][] { };
            var hierarchy = new OpenCvSharp.HierarchyIndex[] { };
            var contourAreaList = new List<double>() { };
            var contourCenterList = new List<System.Drawing.Point>() { };
            Cv2.FindContours(binaryImg, out contours, out hierarchy, RetrievalModes.External, ContourApproximationModes.ApproxNone);
            for (int i = 0; i < hierarchy.Length; i++)
            {
                var M = Cv2.Moments(contours[i]);
                if (M.M00 != 0)
                {
                    var cx = (int)(M.M10 / M.M00);
                    var cy = (int)(M.M01 / M.M00);
                    if (indexer[cy, cx].Item0 == 255)
                    {
                        var rect = Cv2.BoundingRect(contours[i]);

                        if (indexer[rect.Y, rect.X].Item0 == 0 && indexer[rect.Y + rect.Height, rect.X + rect.Width].Item0 == 0
                            && indexer[rect.Y + rect.Height, rect.X].Item0 == 0 && indexer[rect.Y, rect.X + rect.Width].Item0 == 0
                            && rect.Width / rect.Height > 4 && rect.Width > 40)
                        {
                            //Rect r = new Rect(new OpenCvSharp.Point(rect.X, rect.Y), new OpenCvSharp.Size(rect.Width, rect.Height));
                            //Cv2.Rectangle(binaryImg, r, Scalar.LimeGreen, 1);
                            //Cv2.DrawContours(srcImg, contours, i, Scalar.Gray, 2, LineTypes.Link8, hierarchy, 4);
                            var point1 = new OpenCvSharp.Point(cx, cy);

                            Cv2.FloodFill(binaryImg, point1, new Scalar(155, 155, 155));

                            var tempROI = new Mat();
                            binaryImg[rect].CopyTo(tempROI);//截出连通域, 按中心点做漫水填充                            
                            int orientation = 0;
                            ///////////////
                            Vec3b color = new Vec3b(0, 0, 0);
                            int sLeft = 0, sRight = 0;
                            var ROIIndexer = tempROI.GetGenericIndexer<Vec3b>();
                            for (int k = 0; k < tempROI.Rows; k++)
                            {
                                for (int j = 0; j < tempROI.Cols; j++)
                                {
                                    if (ROIIndexer[k, j].Item0 == 255)
                                    {
                                        ROIIndexer[k, j] = color;
                                        continue;
                                    }
                                    else if (ROIIndexer[k, j].Item0 == 155)
                                    {
                                        if (j < tempROI.Cols / 2) sLeft++;
                                        else sRight++;
                                    }
                                }
                            }
                            Cv2.Threshold(tempROI, tempROI, 154, 255, ThresholdTypes.BinaryInv);

                            if (sLeft < sRight)
                            {
                                orientation = 1; //正向箭头
                            }
                            //////////////////
                            var tempContours = new OpenCvSharp.Point[][] { };
                            var tempHierarchy = new OpenCvSharp.HierarchyIndex[] { };
                            Cv2.FindContours(tempROI, out tempContours, out tempHierarchy, RetrievalModes.Tree, ContourApproximationModes.ApproxNone);
                            if (tempContours.GetLength(0) == 4 && Math.Abs(sLeft - sRight) > 10)
                            {
                                rectList.Add(rect);
                                orientationList.Add(orientation);
                            }
                        }
                    }
                }
            }
            return rectList;
        }


        /*//////////////////////////////////////////////第二版本：逐像素手写；主要利用闭运算；处理效率低////////////////////////////
        public List<KeyValuePair<System.Drawing.Point[], int>> GetArrowsSignPostion(Bitmap image)
        {
            var resList = new List<KeyValuePair<System.Drawing.Point[], int>>();
            using (Mat srcImg = OpenCvSharp.Extensions.BitmapConverter.ToMat(image))
            {
                //判断
                bool lowResolution = false;
                int size = 30;
                if (srcImg.Cols < 1000 && srcImg.Cols < 1000)
                {
                    size = 30;
                    lowResolution = true;
                }
                if (srcImg.Cols > 1400 || srcImg.Cols > 1400)
                {
                    size = 60;
                }
                Mat grayImg;
                //Convert input images to gray
                try
                {
                    grayImg = srcImg.CvtColor(ColorConversionCodes.BGR2GRAY);
                }
                catch (OpenCvSharp.OpenCVException)
                {
                    grayImg = srcImg;
                }
                var binaryImg = new Mat();
                Cv2.Threshold(grayImg, binaryImg, 200, 255, ThresholdTypes.Binary);

                var resTempList = new List<System.Drawing.Point[]>();

                InputArray kernel = Cv2.GetStructuringElement(MorphShapes.Rect, new OpenCvSharp.Size(size, 1), new OpenCvSharp.Point(-1, -1));
                var result = new Mat();
                Cv2.MorphologyEx(binaryImg, result, MorphTypes.Close, kernel, iterations: 1);

                //形态化后找到所有的线段坐标并加入到resTempList中
                var lineList = FindLineCoordinate(result, size);
                if (lineList.Count > 200)
                {
                    return resList;
                }
                foreach (var i in lineList)
                {
                    resTempList.Add(i);
                }
                resList = ArrowNMSAndEstimateType(resTempList, binaryImg, lowResolution);
            }
            return resList;
        }

        private List<KeyValuePair<System.Drawing.Point[], int>> ArrowNMSAndEstimateType(List<System.Drawing.Point[]> tempList, Mat binaryImg, bool lowResolution)
        {
            var resList = new List<KeyValuePair<System.Drawing.Point[], int>>();

            var deleteIndexList = new List<int> { };
            tempList.Sort((left, right) => {
                if (left[1].X - left[0].X > right[1].X - right[0].X)
                    return -1;
                else if (left[1].X - left[0].X == right[1].X - right[0].X)
                    return 0;
                else
                    return 1;
            });
            var tempBinaryImg = binaryImg;
            for (int k = 0; k < tempList.Count; k++)
            {
                var flag = false;
                if (deleteIndexList.Contains(k))
                {
                    continue;
                }
                var point1 = new OpenCvSharp.Point(tempList[k][0].X, tempList[k][0].Y);
                //漫水填充
                Cv2.FloodFill(tempBinaryImg, point1, new Scalar(155, 155, 155));

                var indexer = tempBinaryImg.GetGenericIndexer<Vec3b>();
                var pixelList = new List<System.Drawing.Point>();//当前线段连通区域
                for (int i = 0; i < tempBinaryImg.Rows; i++)
                {
                    for (int j = 0; j < tempBinaryImg.Cols; j++)
                    {
                        if (indexer[i, j].Item0 == 155)
                            pixelList.Add(new System.Drawing.Point(j, i));
                    }
                }
                int minX = pixelList[0].X, minY = pixelList[0].Y;//当前连通区域最小ROI
                int maxX = pixelList[0].X, maxY = pixelList[0].Y;
                foreach (var t in pixelList)
                {
                    if (t.X < minX)
                        minX = t.X;
                    else if (t.X > maxX)
                        maxX = t.X;

                    if (t.Y < minY)
                        minY = t.Y;
                    else if (t.Y > maxY)
                        maxY = t.Y;
                }
                //移除待测集里已经被连通的区域，说明这条线段重复了
                for (int i = k + 1; i < tempList.Count; i++)
                {
                    if (indexer[tempList[i][0].Y, tempList[i][0].X].Item0 == 155)
                    {
                        deleteIndexList.Add(i);
                    }
                }
                //长宽比1：4都达不到直接判为不是箭头,移除这一条线段
                if ((maxX - minX) * 1.0 / (maxY - minY) <= 4 || (maxX - minX) < 30)
                {
                    Cv2.FloodFill(tempBinaryImg, point1, new Scalar(0, 0, 0));
                    deleteIndexList.Add(k);
                    continue;
                }
                //裁出当前连通区域,注意去除多余的非连通区域
                var pixelROIMat = new Mat();
                binaryImg[new Rect(new OpenCvSharp.Point(minX, minY), size: new OpenCvSharp.Size(maxX - minX, maxY - minY))].CopyTo(pixelROIMat);
                var ROIIndexer = pixelROIMat.GetGenericIndexer<Vec3b>();
                Vec3b color = new Vec3b(255, 255, 255);
                int sLeft = 0, sRight = 0;
                for (int i = 0; i < pixelROIMat.Rows; i++)
                {
                    for (int j = 0; j < pixelROIMat.Cols; j++)
                    {
                        if (ROIIndexer[i, j].Item0 == 0)
                        {
                            ROIIndexer[i, j] = color;
                            continue;
                        }
                        else if (ROIIndexer[i, j].Item0 == 155)
                        {
                            //统计像素信息，左右像素用于判断箭头类型；
                            if (j < pixelROIMat.Cols / 2) sLeft++;
                            else sRight++;
                        }
                    }
                }
                //左右俩边像素的数量决定箭头类型
                int arrowType = 0;
                if (sLeft > sRight)
                {
                    var t = tempList[k][0];
                    tempList[k][0] = tempList[k][1];
                    tempList[k][1] = t;
                }
                if (lowResolution && Math.Abs(sLeft - sRight) > 5) //如果是低分辨率的图片认为是箭头
                {
                    flag = true;
                    Cv2.FloodFill(tempBinaryImg, point1, new Scalar(0, 0, 0));
                    resList.Add(new KeyValuePair<System.Drawing.Point[], int>(tempList[k], arrowType));
                    continue;
                }

                var contours = new OpenCvSharp.Point[][] { };//如果是高分辨率的图片再通过轮廓判断箭头
                var hierarchy = new OpenCvSharp.HierarchyIndex[] { };
                var temp = new Mat();
                Cv2.Threshold(pixelROIMat, temp, 200, 255, ThresholdTypes.Binary);
                Cv2.FindContours(temp, out contours, out hierarchy, RetrievalModes.Tree, ContourApproximationModes.ApproxNone);
                var contoursCnt = contours.GetLength(0);

                var sList = new List<double>();

                if (contoursCnt == 4)//其实只有单箭头是这样的
                {
                    foreach (var t in contours)
                    {
                        sList.Add(Cv2.ContourArea(t));
                        sList.Sort((n, m) => m.CompareTo(n));
                    }
                    var a = new List<double> { sList[0], sList[2] };
                    var b = new List<double> { sList[1], sList[3] };
                    //上方轮廓的面积对和下方轮廓的面积对 是否相似 //此时符合条件，判断箭头类型
                    var cosineRes = CosineCompare(a, b);
                    if (cosineRes > 0.95)
                    {
                        flag = true;
                        resList.Add(new KeyValuePair<System.Drawing.Point[], int>(tempList[k], arrowType));
                    }
                }
                if (flag == false)
                {
                    deleteIndexList.Add(k);
                }
                Cv2.FloodFill(tempBinaryImg, point1, new Scalar(0, 0, 0));
            }
            return resList;
        }
        private List<System.Drawing.Point[]> FindLineCoordinate(Mat result, int size)
        {
            var resList = new List<System.Drawing.Point[]>();
            var indexer = result.GetGenericIndexer<Vec3b>();
            for (int i = 0; i < result.Rows; i++)
            {
                for (int j = 0; j < result.Cols - size + 1; j++)
                {
                    //j,  j+size-1 
                    int startCol = j;

                    int offset = 0;//偏移必须至少是size-1才是有效的
                    while (true)
                    {
                        if (indexer[i, startCol + offset].Item0 != 0)
                        {
                            break;
                        }
                        offset++;
                    }
                    j += (offset + 1);//无论偏移是否有效，j都要右移，j默认还会+1
                    if (offset < size - 1)
                    {
                        continue;
                    }
                    var resPoint = new System.Drawing.Point[] { new System.Drawing.Point(startCol, i), new System.Drawing.Point(startCol + offset - 2, i) };

                    resList.Add(resPoint);
                }
            }
            return resList;
        }*/


        /*///////////////////////////////////////////////////////第一版：模板匹配，图片分辨率300-3000不等，缺乏尺度不变性/////////////////////////////
        public List<System.Drawing.Point[]> GetArrowsSignPostion(Bitmap image, int scale = 5, double threshold = 0.8)
        {
            var resList = new List<System.Drawing.Point[]>();

            using (Mat refMat = OpenCvSharp.Extensions.BitmapConverter.ToMat(image))
            {

                int refWidth = gref.Width;
                int refHeight = gref.Height;
                int newRefWidth = 1024;
                int newRefHeight = (1024 / refWidth) * refHeight;
                OpenCvSharp.Size a = new OpenCvSharp.Size(refWidth, refHeight);
                Cv2.Resize(refMat, refMat, a);
                //中值滤波
                Cv2.MedianBlur(gref, gref, 5);
                Cv2.Resize(gref, gref, new OpenCvSharp.Size(refWidth, refHeight));

                Mat Tpl = new Mat("arrow_r.png");
                var state = new List<int>() { 1, -1, -2, 2};
                foreach (int rotateState in state)
                {

                    //threshold = 0.783;
                    int srcWidth = Tpl.Width;
                    int srcHeight = Tpl.Height;
                    //rotateState值为1代表正箭头—>,-1代表反箭头<— , -2代表向下箭头，2代表向上箭头
                    var srcTpl = new Mat();
                    switch (rotateState)
                    {
                        case 1:
                            srcTpl = Tpl;
                            break;
                        case -1:
                            srcTpl = new Mat("arrow_l.png");
                            //srcTpl = RotateTpl(Tpl, 180, new OpenCvSharp.Size(srcWidth, srcHeight));

                            break;
                        case -2:
                            srcTpl = new Mat("arrow_d.png");
                            //srcTpl = RotateTpl(Tpl, -90, new OpenCvSharp.Size(srcHeight, srcWidth));

                            break;
                        case 2:
                            srcTpl = new Mat("arrow_u.png");
                            //srcTpl = RotateTpl(Tpl, 90, new OpenCvSharp.Size(200, 200));

                            break;
                        default:
                            break;
                    }

                    //Convert input images to gray
                    Mat gref = refMat.CvtColor(ColorConversionCodes.BGR2GRAY);

                    //Convert templete images to gray
                    Mat gTpl = srcTpl.CvtColor(ColorConversionCodes.BGR2GRAY);

                    for (int i = 1; i <= scale; i++)
                    {
                        int newWidth = i * srcWidth;
                        int newHeight = i * srcHeight;
                        Mat gTplScale = new Mat();
                        Cv2.Resize(gTpl, gTplScale, new OpenCvSharp.Size(newWidth, newHeight));
                        Mat res = new Mat(refMat.Rows - gTplScale.Rows + 1, refMat.Cols - gTplScale.Cols + 1, MatType.CV_32FC1);
                        Cv2.Threshold(res, res, 0.8, 1.0, ThresholdTypes.Tozero);
                        Cv2.MatchTemplate(gref, gTplScale, res, TemplateMatchModes.CCoeffNormed);

                        while (true)
                        {
                            double minval, maxval;
                            OpenCvSharp.Point minloc, maxloc;
                            Cv2.MinMaxLoc(res, out minval, out maxval, out minloc, out maxloc);

                            //把起始点设为匹配到的中心
                            var startPoint = new System.Drawing.Point(maxloc.X + newWidth / 2, maxloc.Y + newHeight / 2);
                            //结束点通过起始点沿像素寻找，为末尾中间点
                            var endPoint = FindEndPointLoc(startPoint, gref, rotateState, gTplScale, gTplScale.Cols);
                            if (rotateState == 1)
                            {
                                //把开始点和结束点交换位置
                                var temp = startPoint;
                                startPoint = endPoint;
                                endPoint = temp;

                                endPoint.X += newWidth / 2;

                            }
                            else if (rotateState == -1)
                            {
                                var temp = startPoint;
                                startPoint = endPoint;
                                endPoint = temp;
                                endPoint.X -= newWidth / 2;

                            }
                            else if (rotateState == -2)
                            {
                                //把开始点和结束点交换位置
                                var temp = startPoint;
                                startPoint = endPoint;
                                endPoint = temp;
                                endPoint.Y += newHeight / 2;

                            }
                            else if (rotateState == 2)
                            {
                                var temp = startPoint;
                                startPoint = endPoint;
                                endPoint = temp;
                                endPoint.Y -= newHeight / 2;

                            }
                            if (maxval >= threshold)
                            {
                                //double standard = Math.Sqrt(Math.Pow(srcWidth, 2) + Math.Pow(srcHeight/2, 2));
                                //把箭头起点和到终点 按顺序压入List
                                resList = UpdateReslist(startPoint, endPoint, resList);
                                Cv2.FloodFill(res, maxloc, new Scalar(0));
                            }
                            else
                                break;
                        }
                    }
                }                
            }
            return resList;
        }

        private System.Drawing.Point FindEndPointLoc(System.Drawing.Point startPoint, Mat gref, int rotateState, Mat gTplScale, int scale = 10)
        {
            //传入的起始点是中心往右/左移动
            //startPoint.X -= rotateState*gTplScale.Cols / 2;
            var endPoint = new System.Drawing.Point(startPoint.X, startPoint.Y);
            Mat binaryImg = new Mat();
            Cv2.Threshold(gref, binaryImg, 200, 255, 0);

            //the arrow template 's width
            int Size = scale;
            InputArray kernel;
            if (rotateState == 2 || rotateState == -2)
                kernel = Cv2.GetStructuringElement(MorphShapes.Rect, new OpenCvSharp.Size(1, Size), new OpenCvSharp.Point(-1, -1));
            else
                kernel = Cv2.GetStructuringElement(MorphShapes.Rect, new OpenCvSharp.Size(Size, 1), new OpenCvSharp.Point(-1, -1));

            Mat result = new Mat();
            //dilate and erode
            Cv2.MorphologyEx(binaryImg, result, MorphTypes.Close, kernel);

            int offset = 0;
            var a = result.At<Vec3b>(startPoint.Y, startPoint.X);
            if (a.Item0 == 0)
            {
                while (true)
                {
                    //断像素，寻找邻域像素
                    //int flag = 0;
                    for (int i = -2; i <= 2; i++)
                    {
                        for (int j = 0; j <= 2; j++)
                        {
                            if (result.At<Vec3b>(startPoint.Y + i, startPoint.X - offset * rotateState + j).Item0 == 0)
                            {
                                flag = 1;
                                break;
                            }
                        }
                        if (flag == 1)
                        {
                            break;
                        }
                    }
                    //水平方向操作
                    if (result.At<Vec3b>(startPoint.Y, startPoint.X - offset * rotateState).Item0 == 0 && (rotateState == 1 || rotateState == -1))
                        offset++;
                    else if (result.At<Vec3b>(startPoint.Y + offset * rotateState / 2, startPoint.X).Item0 == 0 && (rotateState == 2 || rotateState == -2))
                        offset++;
                    else
                    {
                        break;
                    }
                }
            }
            if (rotateState == 2 || rotateState == -2)
                endPoint.Y += offset * rotateState / 2;
            else
                endPoint.X -= offset * rotateState;
            //画框调试
            Rect r = new Rect(new OpenCvSharp.Point(startPoint.X, startPoint.Y), new OpenCvSharp.Size(offset * rotateState, offset * rotateState));
            Cv2.Rectangle(result, r, Scalar.LimeGreen, 1);
            Cv2.ImShow("test", result);
            Cv2.WaitKey(0);
            return endPoint;
        }
        private Mat RotateTpl(Mat srcTpl, double angle, OpenCvSharp.Size newSize)
        {
            Point2f center = new Point2f(srcTpl.Cols / 2, srcTpl.Rows / 2);

            Mat affine_matrix = Cv2.GetRotationMatrix2D(center, angle, 1);//求得旋转矩阵
            Mat newTpl = new Mat();
            Cv2.WarpAffine(srcTpl, newTpl, affine_matrix, newSize);

            return newTpl;
        }
        private bool IsAccept(System.Drawing.Point res, List<System.Drawing.Point> resList, int srcWidth, int srcHeight)
        {
            for (int i = 0; i < resList.Count; i++)
            {
                if (Math.Abs(res.X - resList[i].X) <= srcWidth && Math.Abs(res.Y - resList[i].Y) <= srcHeight)
                {
                    return false;
                }
            }
            return true;
        }
        private List<System.Drawing.Point[]> UpdateReslist(System.Drawing.Point startPoint, System.Drawing.Point endPoint, List<System.Drawing.Point[]> resList, double standard = 10)
        {
            //一个框是startPoint.X  .Y, endPoint.X .Y
            //另一个框是resList[i][0].X .Y， resList[i][1].X  .Y
            bool flag = true;
            int s1 = (endPoint.Y - startPoint.Y) * (endPoint.X - startPoint.X);
            for (int i = 0; i < resList.Count; i++)
            {
                //假如起点或尾点很近，谁面积大保留谁
                int s2 = (resList[i][0].Y - resList[i][1].Y) * (resList[i][0].X - resList[i][1].X);
                double distance1 = Math.Sqrt(Math.Pow(startPoint.X - resList[i][0].X, 2) + Math.Pow(startPoint.Y - resList[i][0].Y, 2));
                double distance2 = Math.Sqrt(Math.Pow(endPoint.X - resList[i][1].X, 2) + Math.Pow(endPoint.Y - resList[i][1].Y, 2));
                if (distance1 <= standard || distance2 <= standard)
                {
                    flag = false;
                    if (s1 > s2)
                    {
                        resList[i][0] = startPoint;
                        resList[i][1] = endPoint;
                        return resList;
                    }
                }

            }
            if (flag && (Math.Abs(endPoint.X - startPoint.X) + Math.Abs(endPoint.Y - startPoint.Y)) > 2 * standard)
            {
                resList.Add(new System.Drawing.Point[] { startPoint, endPoint });
            }
            return resList;
        }*/


        /*/////////////////////////////////////////////第零版：尝试霍夫变换/////////////////////////////
        int newWidth = 1024;
        int newHeight = (1024/src.Width) *src.Height;
        OpenCvSharp.Size a = new OpenCvSharp.Size(newWidth, newHeight); 
        Cv2.Resize(src, src, a);


        //中值滤波
        Cv2.MedianBlur(src, src, 5);
        //阈值化
        int data = 1;
        Cv2.Threshold(src, src, 200, 255, (ThresholdTypes)(data));

        //要找平行方向的线，做竖直方向的开操作（先腐蚀后膨胀）
        int xSize = 100;    //
        InputArray kernelX = Cv2.GetStructuringElement(MorphShapes.Rect, new Size(xSize, 1), new Point(-1, -1));
        Mat result = new Mat();
        Cv2.MorphologyEx(src, result, MorphTypes.Open, kernelX);


        //Canny边缘检测
        Mat cany = new Mat();
        Cv2.Canny(result, cany, 50, 200);

        //霍夫变换
        LineSegmentPoint[] linePoint = Cv2.HoughLinesP(cany, 1, 1, 1, 1,10);
        Scalar color = new Scalar(0, 0, 0);
        Console.WriteLine(linePoint.Count());

        //画图
        Mat dst = new Mat(src.Size(), MatType.CV_8UC3, Scalar.Blue);
        for (int i = 0; i < linePoint.Count(); i++)
        {
            Point p1 = linePoint[i].P1;
            Point p2 = linePoint[i].P2;
            Cv2.Line(dst, p1, p2, color, 1, LineTypes.Link8);
        }
        using (new Window("Hough", WindowMode.AutoSize, dst))
        using (new Window("CANYY", WindowMode.AutoSize, cany))
        using (new Window("SRC", WindowMode.AutoSize, src))
        using (new Window("result", WindowMode.AutoSize, result))
        {
            Cv2.WaitKey(0);
        }*/
    }
}
