function lgraph = l_20

% k = data1();
% disp(size(k));
% x = k(1:100000,:,:);
% y = k(100001:200000,:,:);
digitDatasetPath = fullfile('E:\Matlab convnet examples\project freelancer\dataset\1\');
x = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','none');
% numTrainFiles = 100;
% [xTrain,xValidation] = splitEachLabel(x,0.80);

digitDatasetPath1 = fullfile('E:\Matlab convnet examples\project freelancer\dataset\2\');
y = imageDatastore(digitDatasetPath1, ...
    'IncludeSubfolders',true,'LabelSource','none');
% numTrainFiles = 100;
% [yTrain,yValidation] = splitEachLabel(y,0.80);
% disp(size(readimage(x,1)));
% imdsCombined = combine(x,y);
imdsCombined = randomPatchExtractionDatastore(x,y,[1000 500], ....
'PatchesPerImage',64);

lgraph = layerGraph();

tempLayers = [
    imageInputLayer([1000 500 3],"Name","imageinput")
    convolution2dLayer([1 1],64,"Name","conv_start")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_1_1")
    reluLayer("Name","relu_1_1_1")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_1_1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_1_1")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_1_1")
    reluLayer("Name","relu_2_1_1")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_1_1")
    helperSigmoidLayer("sigmoid_1_1_1","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_1_1",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_1_1");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_2_1")
    reluLayer("Name","relu_1_2_1")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_2_1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_2_1")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_2_1")
    reluLayer("Name","relu_2_2_1")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_2_1")
    helperSigmoidLayer("sigmoid_1_2_1","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_2_1",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_2_1");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_3_1")
    reluLayer("Name","relu_1_3_1")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_3_1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_3_1")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_3_1")
    reluLayer("Name","relu_2_3_1")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_3_1")
    helperSigmoidLayer("sigmoid_1_3_1","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_3_1",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_3_1");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_4_1")
    reluLayer("Name","relu_1_4_1")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_4_1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_4_1")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_4_1")
    reluLayer("Name","relu_2_4_1")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_4_1")
    helperSigmoidLayer("sigmoid_1_4_1","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_4_1",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_4_1");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_5_1")
    reluLayer("Name","relu_1_5_1")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_5_1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_5_1")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_5_1")
    reluLayer("Name","relu_2_5_1")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_5_1")
    helperSigmoidLayer("sigmoid_1_5_1","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_5_1",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_5_1");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_6_1")
    reluLayer("Name","relu_1_6_1")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_6_1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_6_1")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_6_1")
    reluLayer("Name","relu_2_6_1")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_6_1")
    helperSigmoidLayer("sigmoid_1_6_1","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_6_1",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_6_1");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_7_1")
    reluLayer("Name","relu_1_7_1")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_7_1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_7_1")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_7_1")
    reluLayer("Name","relu_2_7_1")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_7_1")
    helperSigmoidLayer("sigmoid_1_7_1","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_7_1",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_7_1");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_8_1")
    reluLayer("Name","relu_1_8_1")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_8_1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_8_1")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_8_1")
    reluLayer("Name","relu_2_8_1")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_8_1")
    helperSigmoidLayer("sigmoid_1_8_1","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_8_1",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_8_1");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_9_1")
    reluLayer("Name","relu_1_9_1")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_9_1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_9_1")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_9_1")
    reluLayer("Name","relu_2_9_1")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_9_1")
    helperSigmoidLayer("sigmoid_1_9_1","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_9_1",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_9_1");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_10_1")
    reluLayer("Name","relu_1_10_1")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_10_1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_10_1")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_10_1")
    reluLayer("Name","relu_2_10_1")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_10_1")
    helperSigmoidLayer("sigmoid_1_10_1","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_10_1",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_10_1");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_11_1")
    reluLayer("Name","relu_1_11_1")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_11_1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_11_1")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_11_1")
    reluLayer("Name","relu_2_11_1")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_11_1")
    helperSigmoidLayer("sigmoid_1_11_1","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_11_1",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_11_1");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_12_1")
    reluLayer("Name","relu_1_12_1")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_12_1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_12_1")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_12_1")
    reluLayer("Name","relu_2_12_1")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_12_1")
    helperSigmoidLayer("sigmoid_1_12_1","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_12_1",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_12_1");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_13_1")
    reluLayer("Name","relu_1_13_1")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_13_1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_13_1")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_13_1")
    reluLayer("Name","relu_2_13_1")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_13_1")
    helperSigmoidLayer("sigmoid_1_13_1","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_13_1",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_13_1");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_14_1")
    reluLayer("Name","relu_1_14_1")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_14_1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_14_1")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_14_1")
    reluLayer("Name","relu_2_14_1")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_14_1")
    helperSigmoidLayer("sigmoid_1_14_1","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_14_1",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_14_1");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_15_1")
    reluLayer("Name","relu_1_15_1")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_15_1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_15_1")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_15_1")
    reluLayer("Name","relu_2_15_1")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_15_1")
    helperSigmoidLayer("sigmoid_1_15_1","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_15_1",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_15_1");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_16_1")
    reluLayer("Name","relu_1_16_1")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_16_1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_16_1")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_16_1")
    reluLayer("Name","relu_2_16_1")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_16_1")
    helperSigmoidLayer("sigmoid_1_16_1","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_16_1",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_16_1");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_17_1")
    reluLayer("Name","relu_1_17_1")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_17_1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_17_1")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_17_1")
    reluLayer("Name","relu_2_17_1")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_17_1")
    helperSigmoidLayer("sigmoid_1_17_1","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_17_1",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_17_1");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_18_1")
    reluLayer("Name","relu_1_18_1")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_18_1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_18_1")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_18_1")
    reluLayer("Name","relu_2_18_1")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_18_1")
    helperSigmoidLayer("sigmoid_1_18_1","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_18_1",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_18_1");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_19_1")
    reluLayer("Name","relu_1_19_1")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_19_1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_19_1")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_19_1")
    reluLayer("Name","relu_2_19_1")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_19_1")
    helperSigmoidLayer("sigmoid_1_19_1","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_19_1",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_19_1");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_20_1")
    reluLayer("Name","relu_1_20_1")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_20_1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_20_1")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_20_1")
    reluLayer("Name","relu_2_20_1")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_20_1")
    helperSigmoidLayer("sigmoid_1_20_1","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_20_1",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","addition_20_1")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1")];
lgraph = addLayers(lgraph,tempLayers);
l = [additionLayer(2,'Name','sum_1')];
lgraph = addLayers(lgraph,l);
lgraph = connectLayers(lgraph,'conv_start','sum_1/in1');
lgraph = connectLayers(lgraph,'conv_1','sum_1/in2');
lgraph = connectLayers(lgraph,"conv_start","conv_1_1_1");
lgraph = connectLayers(lgraph,"conv_start","addition_1_1/in2");
lgraph = connectLayers(lgraph,"conv_2_1_1","maxpool_1_1");
lgraph = connectLayers(lgraph,"conv_2_1_1","mul_1_1_1/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_1_1","mul_1_1_1/in1");
lgraph = connectLayers(lgraph,"mul_1_1_1","addition_1_1/in1");
lgraph = connectLayers(lgraph,"addition_1_1","conv_1_2_1");
lgraph = connectLayers(lgraph,"addition_1_1","addition_2_1/in2");
lgraph = connectLayers(lgraph,"conv_2_2_1","maxpool_2_1");
lgraph = connectLayers(lgraph,"conv_2_2_1","mul_1_2_1/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_2_1","mul_1_2_1/in1");
lgraph = connectLayers(lgraph,"mul_1_2_1","addition_2_1/in1");
lgraph = connectLayers(lgraph,"addition_2_1","conv_1_3_1");
lgraph = connectLayers(lgraph,"addition_2_1","addition_3_1/in2");
lgraph = connectLayers(lgraph,"conv_2_3_1","maxpool_3_1");
lgraph = connectLayers(lgraph,"conv_2_3_1","mul_1_3_1/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_3_1","mul_1_3_1/in1");
lgraph = connectLayers(lgraph,"mul_1_3_1","addition_3_1/in1");
lgraph = connectLayers(lgraph,"addition_3_1","conv_1_4_1");
lgraph = connectLayers(lgraph,"addition_3_1","addition_4_1/in2");
lgraph = connectLayers(lgraph,"conv_2_4_1","maxpool_4_1");
lgraph = connectLayers(lgraph,"conv_2_4_1","mul_1_4_1/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_4_1","mul_1_4_1/in1");
lgraph = connectLayers(lgraph,"mul_1_4_1","addition_4_1/in1");
lgraph = connectLayers(lgraph,"addition_4_1","conv_1_5_1");
lgraph = connectLayers(lgraph,"addition_4_1","addition_5_1/in2");
lgraph = connectLayers(lgraph,"conv_2_5_1","maxpool_5_1");
lgraph = connectLayers(lgraph,"conv_2_5_1","mul_1_5_1/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_5_1","mul_1_5_1/in1");
lgraph = connectLayers(lgraph,"mul_1_5_1","addition_5_1/in1");
lgraph = connectLayers(lgraph,"addition_5_1","conv_1_6_1");
lgraph = connectLayers(lgraph,"addition_5_1","addition_6_1/in2");
lgraph = connectLayers(lgraph,"conv_2_6_1","maxpool_6_1");
lgraph = connectLayers(lgraph,"conv_2_6_1","mul_1_6_1/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_6_1","mul_1_6_1/in1");
lgraph = connectLayers(lgraph,"mul_1_6_1","addition_6_1/in1");
lgraph = connectLayers(lgraph,"addition_6_1","conv_1_7_1");
lgraph = connectLayers(lgraph,"addition_6_1","addition_7_1/in2");
lgraph = connectLayers(lgraph,"conv_2_7_1","maxpool_7_1");
lgraph = connectLayers(lgraph,"conv_2_7_1","mul_1_7_1/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_7_1","mul_1_7_1/in1");
lgraph = connectLayers(lgraph,"mul_1_7_1","addition_7_1/in1");
lgraph = connectLayers(lgraph,"addition_7_1","conv_1_8_1");
lgraph = connectLayers(lgraph,"addition_7_1","addition_8_1/in2");
lgraph = connectLayers(lgraph,"conv_2_8_1","maxpool_8_1");
lgraph = connectLayers(lgraph,"conv_2_8_1","mul_1_8_1/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_8_1","mul_1_8_1/in1");
lgraph = connectLayers(lgraph,"mul_1_8_1","addition_8_1/in1");
lgraph = connectLayers(lgraph,"addition_8_1","conv_1_9_1");
lgraph = connectLayers(lgraph,"addition_8_1","addition_9_1/in2");
lgraph = connectLayers(lgraph,"conv_2_9_1","maxpool_9_1");
lgraph = connectLayers(lgraph,"conv_2_9_1","mul_1_9_1/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_9_1","mul_1_9_1/in1");
lgraph = connectLayers(lgraph,"mul_1_9_1","addition_9_1/in1");
lgraph = connectLayers(lgraph,"addition_9_1","conv_1_10_1");
lgraph = connectLayers(lgraph,"addition_9_1","addition_10_1/in2");
lgraph = connectLayers(lgraph,"conv_2_10_1","maxpool_10_1");
lgraph = connectLayers(lgraph,"conv_2_10_1","mul_1_10_1/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_10_1","mul_1_10_1/in1");
lgraph = connectLayers(lgraph,"mul_1_10_1","addition_10_1/in1");
lgraph = connectLayers(lgraph,"addition_10_1","conv_1_11_1");
lgraph = connectLayers(lgraph,"addition_10_1","addition_11_1/in2");
lgraph = connectLayers(lgraph,"conv_2_11_1","maxpool_11_1");
lgraph = connectLayers(lgraph,"conv_2_11_1","mul_1_11_1/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_11_1","mul_1_11_1/in1");
lgraph = connectLayers(lgraph,"mul_1_11_1","addition_11_1/in1");
lgraph = connectLayers(lgraph,"addition_11_1","conv_1_12_1");
lgraph = connectLayers(lgraph,"addition_11_1","addition_12_1/in2");
lgraph = connectLayers(lgraph,"conv_2_12_1","maxpool_12_1");
lgraph = connectLayers(lgraph,"conv_2_12_1","mul_1_12_1/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_12_1","mul_1_12_1/in1");
lgraph = connectLayers(lgraph,"mul_1_12_1","addition_12_1/in1");
lgraph = connectLayers(lgraph,"addition_12_1","conv_1_13_1");
lgraph = connectLayers(lgraph,"addition_12_1","addition_13_1/in2");
lgraph = connectLayers(lgraph,"conv_2_13_1","maxpool_13_1");
lgraph = connectLayers(lgraph,"conv_2_13_1","mul_1_13_1/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_13_1","mul_1_13_1/in1");
lgraph = connectLayers(lgraph,"mul_1_13_1","addition_13_1/in1");
lgraph = connectLayers(lgraph,"addition_13_1","conv_1_14_1");
lgraph = connectLayers(lgraph,"addition_13_1","addition_14_1/in2");
lgraph = connectLayers(lgraph,"conv_2_14_1","maxpool_14_1");
lgraph = connectLayers(lgraph,"conv_2_14_1","mul_1_14_1/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_14_1","mul_1_14_1/in1");
lgraph = connectLayers(lgraph,"mul_1_14_1","addition_14_1/in1");
lgraph = connectLayers(lgraph,"addition_14_1","conv_1_15_1");
lgraph = connectLayers(lgraph,"addition_14_1","addition_15_1/in2");
lgraph = connectLayers(lgraph,"conv_2_15_1","maxpool_15_1");
lgraph = connectLayers(lgraph,"conv_2_15_1","mul_1_15_1/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_15_1","mul_1_15_1/in1");
lgraph = connectLayers(lgraph,"mul_1_15_1","addition_15_1/in1");
lgraph = connectLayers(lgraph,"addition_15_1","conv_1_16_1");
lgraph = connectLayers(lgraph,"addition_15_1","addition_16_1/in2");
lgraph = connectLayers(lgraph,"conv_2_16_1","maxpool_16_1");
lgraph = connectLayers(lgraph,"conv_2_16_1","mul_1_16_1/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_16_1","mul_1_16_1/in1");
lgraph = connectLayers(lgraph,"mul_1_16_1","addition_16_1/in1");
lgraph = connectLayers(lgraph,"addition_16_1","conv_1_17_1");
lgraph = connectLayers(lgraph,"addition_16_1","addition_17_1/in2");
lgraph = connectLayers(lgraph,"conv_2_17_1","maxpool_17_1");
lgraph = connectLayers(lgraph,"conv_2_17_1","mul_1_17_1/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_17_1","mul_1_17_1/in1");
lgraph = connectLayers(lgraph,"mul_1_17_1","addition_17_1/in1");
lgraph = connectLayers(lgraph,"addition_17_1","conv_1_18_1");
lgraph = connectLayers(lgraph,"addition_17_1","addition_18_1/in2");
lgraph = connectLayers(lgraph,"conv_2_18_1","maxpool_18_1");
lgraph = connectLayers(lgraph,"conv_2_18_1","mul_1_18_1/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_18_1","mul_1_18_1/in1");
lgraph = connectLayers(lgraph,"mul_1_18_1","addition_18_1/in1");
lgraph = connectLayers(lgraph,"addition_18_1","conv_1_19_1");
lgraph = connectLayers(lgraph,"addition_18_1","addition_19_1/in2");
lgraph = connectLayers(lgraph,"conv_2_19_1","maxpool_19_1");
lgraph = connectLayers(lgraph,"conv_2_19_1","mul_1_19_1/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_19_1","mul_1_19_1/in1");
lgraph = connectLayers(lgraph,"mul_1_19_1","addition_19_1/in1");
lgraph = connectLayers(lgraph,"addition_19_1","conv_1_20_1");
lgraph = connectLayers(lgraph,"addition_19_1","addition_20_1/in2");
lgraph = connectLayers(lgraph,"conv_2_20_1","maxpool_20_1");
lgraph = connectLayers(lgraph,"conv_2_20_1","mul_1_20_1/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_20_1","mul_1_20_1/in1");
lgraph = connectLayers(lgraph,"mul_1_20_1","addition_20_1/in1");

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_1_2")
    reluLayer("Name","relu_1_1_2")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_1_2")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_1_2")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_1_2")
    reluLayer("Name","relu_2_1_2")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_1_2")
    helperSigmoidLayer("sigmoid_1_1_2","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_1_2",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_1_2");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_2_2")
    reluLayer("Name","relu_1_2_2")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_2_2")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_2_2")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_2_2")
    reluLayer("Name","relu_2_2_2")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_2_2")
    helperSigmoidLayer("sigmoid_1_2_2","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_2_2",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_2_2");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_3_2")
    reluLayer("Name","relu_1_3_2")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_3_2")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_3_2")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_3_2")
    reluLayer("Name","relu_2_3_2")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_3_2")
    helperSigmoidLayer("sigmoid_1_3_2","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_3_2",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_3_2");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_4_2")
    reluLayer("Name","relu_1_4_2")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_4_2")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_4_2")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_4_2")
    reluLayer("Name","relu_2_4_2")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_4_2")
    helperSigmoidLayer("sigmoid_1_4_2","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_4_2",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_4_2");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_5_2")
    reluLayer("Name","relu_1_5_2")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_5_2")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_5_2")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_5_2")
    reluLayer("Name","relu_2_5_2")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_5_2")
    helperSigmoidLayer("sigmoid_1_5_2","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_5_2",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_5_2");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_6_2")
    reluLayer("Name","relu_1_6_2")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_6_2")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_6_2")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_6_2")
    reluLayer("Name","relu_2_6_2")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_6_2")
    helperSigmoidLayer("sigmoid_1_6_2","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_6_2",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_6_2");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_7_2")
    reluLayer("Name","relu_1_7_2")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_7_2")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_7_2")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_7_2")
    reluLayer("Name","relu_2_7_2")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_7_2")
    helperSigmoidLayer("sigmoid_1_7_2","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_7_2",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_7_2");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_8_2")
    reluLayer("Name","relu_1_8_2")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_8_2")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_8_2")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_8_2")
    reluLayer("Name","relu_2_8_2")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_8_2")
    helperSigmoidLayer("sigmoid_1_8_2","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_8_2",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_8_2");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_9_2")
    reluLayer("Name","relu_1_9_2")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_9_2")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_9_2")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_9_2")
    reluLayer("Name","relu_2_9_2")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_9_2")
    helperSigmoidLayer("sigmoid_1_9_2","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_9_2",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_9_2");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_10_2")
    reluLayer("Name","relu_1_10_2")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_10_2")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_10_2")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_10_2")
    reluLayer("Name","relu_2_10_2")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_10_2")
    helperSigmoidLayer("sigmoid_1_10_2","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_10_2",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_10_2");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_11_2")
    reluLayer("Name","relu_1_11_2")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_11_2")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_11_2")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_11_2")
    reluLayer("Name","relu_2_11_2")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_11_2")
    helperSigmoidLayer("sigmoid_1_11_2","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_11_2",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_11_2");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_12_2")
    reluLayer("Name","relu_1_12_2")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_12_2")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_12_2")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_12_2")
    reluLayer("Name","relu_2_12_2")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_12_2")
    helperSigmoidLayer("sigmoid_1_12_2","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_12_2",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_12_2");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_13_2")
    reluLayer("Name","relu_1_13_2")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_13_2")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_13_2")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_13_2")
    reluLayer("Name","relu_2_13_2")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_13_2")
    helperSigmoidLayer("sigmoid_1_13_2","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_13_2",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_13_2");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_14_2")
    reluLayer("Name","relu_1_14_2")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_14_2")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_14_2")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_14_2")
    reluLayer("Name","relu_2_14_2")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_14_2")
    helperSigmoidLayer("sigmoid_1_14_2","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_14_2",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_14_2");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_15_2")
    reluLayer("Name","relu_1_15_2")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_15_2")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_15_2")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_15_2")
    reluLayer("Name","relu_2_15_2")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_15_2")
    helperSigmoidLayer("sigmoid_1_15_2","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_15_2",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_15_2");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_16_2")
    reluLayer("Name","relu_1_16_2")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_16_2")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_16_2")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_16_2")
    reluLayer("Name","relu_2_16_2")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_16_2")
    helperSigmoidLayer("sigmoid_1_16_2","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_16_2",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_16_2");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_17_2")
    reluLayer("Name","relu_1_17_2")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_17_2")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_17_2")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_17_2")
    reluLayer("Name","relu_2_17_2")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_17_2")
    helperSigmoidLayer("sigmoid_1_17_2","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_17_2",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_17_2");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_18_2")
    reluLayer("Name","relu_1_18_2")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_18_2")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_18_2")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_18_2")
    reluLayer("Name","relu_2_18_2")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_18_2")
    helperSigmoidLayer("sigmoid_1_18_2","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_18_2",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_18_2");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_19_2")
    reluLayer("Name","relu_1_19_2")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_19_2")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_19_2")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_19_2")
    reluLayer("Name","relu_2_19_2")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_19_2")
    helperSigmoidLayer("sigmoid_1_19_2","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_19_2",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_19_2");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_20_2")
    reluLayer("Name","relu_1_20_2")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_20_2")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_20_2")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_20_2")
    reluLayer("Name","relu_2_20_2")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_20_2")
    helperSigmoidLayer("sigmoid_1_20_2","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_20_2",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","addition_20_2")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2")];
lgraph = addLayers(lgraph,tempLayers);
l = [additionLayer(2,'Name','sum_1_2')];
lgraph = addLayers(lgraph,l);
lgraph = connectLayers(lgraph,'sum_1','sum_1_2/in1');
lgraph = connectLayers(lgraph,'conv_2','sum_1_2/in2');
lgraph = connectLayers(lgraph,"sum_1","conv_1_1_2");
lgraph = connectLayers(lgraph,"sum_1","addition_1_2/in2");
lgraph = connectLayers(lgraph,"conv_2_1_2","maxpool_1_2");
lgraph = connectLayers(lgraph,"conv_2_1_2","mul_1_1_2/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_1_2","mul_1_1_2/in1");
lgraph = connectLayers(lgraph,"mul_1_1_2","addition_1_2/in1");
lgraph = connectLayers(lgraph,"addition_1_2","conv_1_2_2");
lgraph = connectLayers(lgraph,"addition_1_2","addition_2_2/in2");
lgraph = connectLayers(lgraph,"conv_2_2_2","maxpool_2_2");
lgraph = connectLayers(lgraph,"conv_2_2_2","mul_1_2_2/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_2_2","mul_1_2_2/in1");
lgraph = connectLayers(lgraph,"mul_1_2_2","addition_2_2/in1");
lgraph = connectLayers(lgraph,"addition_2_2","conv_1_3_2");
lgraph = connectLayers(lgraph,"addition_2_2","addition_3_2/in2");
lgraph = connectLayers(lgraph,"conv_2_3_2","maxpool_3_2");
lgraph = connectLayers(lgraph,"conv_2_3_2","mul_1_3_2/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_3_2","mul_1_3_2/in1");
lgraph = connectLayers(lgraph,"mul_1_3_2","addition_3_2/in1");
lgraph = connectLayers(lgraph,"addition_3_2","conv_1_4_2");
lgraph = connectLayers(lgraph,"addition_3_2","addition_4_2/in2");
lgraph = connectLayers(lgraph,"conv_2_4_2","maxpool_4_2");
lgraph = connectLayers(lgraph,"conv_2_4_2","mul_1_4_2/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_4_2","mul_1_4_2/in1");
lgraph = connectLayers(lgraph,"mul_1_4_2","addition_4_2/in1");
lgraph = connectLayers(lgraph,"addition_4_2","conv_1_5_2");
lgraph = connectLayers(lgraph,"addition_4_2","addition_5_2/in2");
lgraph = connectLayers(lgraph,"conv_2_5_2","maxpool_5_2");
lgraph = connectLayers(lgraph,"conv_2_5_2","mul_1_5_2/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_5_2","mul_1_5_2/in1");
lgraph = connectLayers(lgraph,"mul_1_5_2","addition_5_2/in1");
lgraph = connectLayers(lgraph,"addition_5_2","conv_1_6_2");
lgraph = connectLayers(lgraph,"addition_5_2","addition_6_2/in2");
lgraph = connectLayers(lgraph,"conv_2_6_2","maxpool_6_2");
lgraph = connectLayers(lgraph,"conv_2_6_2","mul_1_6_2/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_6_2","mul_1_6_2/in1");
lgraph = connectLayers(lgraph,"mul_1_6_2","addition_6_2/in1");
lgraph = connectLayers(lgraph,"addition_6_2","conv_1_7_2");
lgraph = connectLayers(lgraph,"addition_6_2","addition_7_2/in2");
lgraph = connectLayers(lgraph,"conv_2_7_2","maxpool_7_2");
lgraph = connectLayers(lgraph,"conv_2_7_2","mul_1_7_2/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_7_2","mul_1_7_2/in1");
lgraph = connectLayers(lgraph,"mul_1_7_2","addition_7_2/in1");
lgraph = connectLayers(lgraph,"addition_7_2","conv_1_8_2");
lgraph = connectLayers(lgraph,"addition_7_2","addition_8_2/in2");
lgraph = connectLayers(lgraph,"conv_2_8_2","maxpool_8_2");
lgraph = connectLayers(lgraph,"conv_2_8_2","mul_1_8_2/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_8_2","mul_1_8_2/in1");
lgraph = connectLayers(lgraph,"mul_1_8_2","addition_8_2/in1");
lgraph = connectLayers(lgraph,"addition_8_2","conv_1_9_2");
lgraph = connectLayers(lgraph,"addition_8_2","addition_9_2/in2");
lgraph = connectLayers(lgraph,"conv_2_9_2","maxpool_9_2");
lgraph = connectLayers(lgraph,"conv_2_9_2","mul_1_9_2/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_9_2","mul_1_9_2/in1");
lgraph = connectLayers(lgraph,"mul_1_9_2","addition_9_2/in1");
lgraph = connectLayers(lgraph,"addition_9_2","conv_1_10_2");
lgraph = connectLayers(lgraph,"addition_9_2","addition_10_2/in2");
lgraph = connectLayers(lgraph,"conv_2_10_2","maxpool_10_2");
lgraph = connectLayers(lgraph,"conv_2_10_2","mul_1_10_2/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_10_2","mul_1_10_2/in1");
lgraph = connectLayers(lgraph,"mul_1_10_2","addition_10_2/in1");
lgraph = connectLayers(lgraph,"addition_10_2","conv_1_11_2");
lgraph = connectLayers(lgraph,"addition_10_2","addition_11_2/in2");
lgraph = connectLayers(lgraph,"conv_2_11_2","maxpool_11_2");
lgraph = connectLayers(lgraph,"conv_2_11_2","mul_1_11_2/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_11_2","mul_1_11_2/in1");
lgraph = connectLayers(lgraph,"mul_1_11_2","addition_11_2/in1");
lgraph = connectLayers(lgraph,"addition_11_2","conv_1_12_2");
lgraph = connectLayers(lgraph,"addition_11_2","addition_12_2/in2");
lgraph = connectLayers(lgraph,"conv_2_12_2","maxpool_12_2");
lgraph = connectLayers(lgraph,"conv_2_12_2","mul_1_12_2/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_12_2","mul_1_12_2/in1");
lgraph = connectLayers(lgraph,"mul_1_12_2","addition_12_2/in1");
lgraph = connectLayers(lgraph,"addition_12_2","conv_1_13_2");
lgraph = connectLayers(lgraph,"addition_12_2","addition_13_2/in2");
lgraph = connectLayers(lgraph,"conv_2_13_2","maxpool_13_2");
lgraph = connectLayers(lgraph,"conv_2_13_2","mul_1_13_2/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_13_2","mul_1_13_2/in1");
lgraph = connectLayers(lgraph,"mul_1_13_2","addition_13_2/in1");
lgraph = connectLayers(lgraph,"addition_13_2","conv_1_14_2");
lgraph = connectLayers(lgraph,"addition_13_2","addition_14_2/in2");
lgraph = connectLayers(lgraph,"conv_2_14_2","maxpool_14_2");
lgraph = connectLayers(lgraph,"conv_2_14_2","mul_1_14_2/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_14_2","mul_1_14_2/in1");
lgraph = connectLayers(lgraph,"mul_1_14_2","addition_14_2/in1");
lgraph = connectLayers(lgraph,"addition_14_2","conv_1_15_2");
lgraph = connectLayers(lgraph,"addition_14_2","addition_15_2/in2");
lgraph = connectLayers(lgraph,"conv_2_15_2","maxpool_15_2");
lgraph = connectLayers(lgraph,"conv_2_15_2","mul_1_15_2/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_15_2","mul_1_15_2/in1");
lgraph = connectLayers(lgraph,"mul_1_15_2","addition_15_2/in1");
lgraph = connectLayers(lgraph,"addition_15_2","conv_1_16_2");
lgraph = connectLayers(lgraph,"addition_15_2","addition_16_2/in2");
lgraph = connectLayers(lgraph,"conv_2_16_2","maxpool_16_2");
lgraph = connectLayers(lgraph,"conv_2_16_2","mul_1_16_2/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_16_2","mul_1_16_2/in1");
lgraph = connectLayers(lgraph,"mul_1_16_2","addition_16_2/in1");
lgraph = connectLayers(lgraph,"addition_16_2","conv_1_17_2");
lgraph = connectLayers(lgraph,"addition_16_2","addition_17_2/in2");
lgraph = connectLayers(lgraph,"conv_2_17_2","maxpool_17_2");
lgraph = connectLayers(lgraph,"conv_2_17_2","mul_1_17_2/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_17_2","mul_1_17_2/in1");
lgraph = connectLayers(lgraph,"mul_1_17_2","addition_17_2/in1");
lgraph = connectLayers(lgraph,"addition_17_2","conv_1_18_2");
lgraph = connectLayers(lgraph,"addition_17_2","addition_18_2/in2");
lgraph = connectLayers(lgraph,"conv_2_18_2","maxpool_18_2");
lgraph = connectLayers(lgraph,"conv_2_18_2","mul_1_18_2/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_18_2","mul_1_18_2/in1");
lgraph = connectLayers(lgraph,"mul_1_18_2","addition_18_2/in1");
lgraph = connectLayers(lgraph,"addition_18_2","conv_1_19_2");
lgraph = connectLayers(lgraph,"addition_18_2","addition_19_2/in2");
lgraph = connectLayers(lgraph,"conv_2_19_2","maxpool_19_2");
lgraph = connectLayers(lgraph,"conv_2_19_2","mul_1_19_2/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_19_2","mul_1_19_2/in1");
lgraph = connectLayers(lgraph,"mul_1_19_2","addition_19_2/in1");
lgraph = connectLayers(lgraph,"addition_19_2","conv_1_20_2");
lgraph = connectLayers(lgraph,"addition_19_2","addition_20_2/in2");
lgraph = connectLayers(lgraph,"conv_2_20_2","maxpool_20_2");
lgraph = connectLayers(lgraph,"conv_2_20_2","mul_1_20_2/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_20_2","mul_1_20_2/in1");
lgraph = connectLayers(lgraph,"mul_1_20_2","addition_20_2/in1");

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_1_3")
    reluLayer("Name","relu_1_1_3")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_1_3")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_1_3")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_1_3")
    reluLayer("Name","relu_2_1_3")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_1_3")
    helperSigmoidLayer("sigmoid_1_1_3","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_1_3",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_1_3");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_2_3")
    reluLayer("Name","relu_1_2_3")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_2_3")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_2_3")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_2_3")
    reluLayer("Name","relu_2_2_3")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_2_3")
    helperSigmoidLayer("sigmoid_1_2_3","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_2_3",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_2_3");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_3_3")
    reluLayer("Name","relu_1_3_3")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_3_3")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_3_3")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_3_3")
    reluLayer("Name","relu_2_3_3")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_3_3")
    helperSigmoidLayer("sigmoid_1_3_3","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_3_3",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_3_3");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_4_3")
    reluLayer("Name","relu_1_4_3")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_4_3")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_4_3")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_4_3")
    reluLayer("Name","relu_2_4_3")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_4_3")
    helperSigmoidLayer("sigmoid_1_4_3","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_4_3",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_4_3");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_5_3")
    reluLayer("Name","relu_1_5_3")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_5_3")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_5_3")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_5_3")
    reluLayer("Name","relu_2_5_3")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_5_3")
    helperSigmoidLayer("sigmoid_1_5_3","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_5_3",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_5_3");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_6_3")
    reluLayer("Name","relu_1_6_3")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_6_3")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_6_3")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_6_3")
    reluLayer("Name","relu_2_6_3")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_6_3")
    helperSigmoidLayer("sigmoid_1_6_3","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_6_3",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_6_3");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_7_3")
    reluLayer("Name","relu_1_7_3")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_7_3")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_7_3")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_7_3")
    reluLayer("Name","relu_2_7_3")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_7_3")
    helperSigmoidLayer("sigmoid_1_7_3","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_7_3",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_7_3");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_8_3")
    reluLayer("Name","relu_1_8_3")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_8_3")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_8_3")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_8_3")
    reluLayer("Name","relu_2_8_3")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_8_3")
    helperSigmoidLayer("sigmoid_1_8_3","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_8_3",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_8_3");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_9_3")
    reluLayer("Name","relu_1_9_3")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_9_3")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_9_3")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_9_3")
    reluLayer("Name","relu_2_9_3")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_9_3")
    helperSigmoidLayer("sigmoid_1_9_3","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_9_3",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_9_3");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_10_3")
    reluLayer("Name","relu_1_10_3")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_10_3")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_10_3")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_10_3")
    reluLayer("Name","relu_2_10_3")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_10_3")
    helperSigmoidLayer("sigmoid_1_10_3","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_10_3",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_10_3");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_11_3")
    reluLayer("Name","relu_1_11_3")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_11_3")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_11_3")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_11_3")
    reluLayer("Name","relu_2_11_3")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_11_3")
    helperSigmoidLayer("sigmoid_1_11_3","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_11_3",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_11_3");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_12_3")
    reluLayer("Name","relu_1_12_3")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_12_3")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_12_3")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_12_3")
    reluLayer("Name","relu_2_12_3")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_12_3")
    helperSigmoidLayer("sigmoid_1_12_3","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_12_3",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_12_3");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_13_3")
    reluLayer("Name","relu_1_13_3")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_13_3")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_13_3")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_13_3")
    reluLayer("Name","relu_2_13_3")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_13_3")
    helperSigmoidLayer("sigmoid_1_13_3","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_13_3",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_13_3");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_14_3")
    reluLayer("Name","relu_1_14_3")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_14_3")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_14_3")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_14_3")
    reluLayer("Name","relu_2_14_3")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_14_3")
    helperSigmoidLayer("sigmoid_1_14_3","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_14_3",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_14_3");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_15_3")
    reluLayer("Name","relu_1_15_3")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_15_3")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_15_3")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_15_3")
    reluLayer("Name","relu_2_15_3")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_15_3")
    helperSigmoidLayer("sigmoid_1_15_3","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_15_3",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_15_3");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_16_3")
    reluLayer("Name","relu_1_16_3")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_16_3")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_16_3")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_16_3")
    reluLayer("Name","relu_2_16_3")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_16_3")
    helperSigmoidLayer("sigmoid_1_16_3","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_16_3",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_16_3");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_17_3")
    reluLayer("Name","relu_1_17_3")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_17_3")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_17_3")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_17_3")
    reluLayer("Name","relu_2_17_3")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_17_3")
    helperSigmoidLayer("sigmoid_1_17_3","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_17_3",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_17_3");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_18_3")
    reluLayer("Name","relu_1_18_3")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_18_3")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_18_3")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_18_3")
    reluLayer("Name","relu_2_18_3")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_18_3")
    helperSigmoidLayer("sigmoid_1_18_3","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_18_3",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_18_3");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_19_3")
    reluLayer("Name","relu_1_19_3")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_19_3")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_19_3")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_19_3")
    reluLayer("Name","relu_2_19_3")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_19_3")
    helperSigmoidLayer("sigmoid_1_19_3","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_19_3",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_19_3");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_20_3")
    reluLayer("Name","relu_1_20_3")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_20_3")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_20_3")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_20_3")
    reluLayer("Name","relu_2_20_3")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_20_3")
    helperSigmoidLayer("sigmoid_1_20_3","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_20_3",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","addition_20_3")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3")];
lgraph = addLayers(lgraph,tempLayers);
l = [additionLayer(2,'Name','sum_1_3')];
lgraph = addLayers(lgraph,l);
lgraph = connectLayers(lgraph,'sum_1_2','sum_1_3/in1');
lgraph = connectLayers(lgraph,'conv_3','sum_1_3/in2');
lgraph = connectLayers(lgraph,"sum_1_2","conv_1_1_3");
lgraph = connectLayers(lgraph,"sum_1_2","addition_1_3/in2");
lgraph = connectLayers(lgraph,"conv_2_1_3","maxpool_1_3");
lgraph = connectLayers(lgraph,"conv_2_1_3","mul_1_1_3/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_1_3","mul_1_1_3/in1");
lgraph = connectLayers(lgraph,"mul_1_1_3","addition_1_3/in1");
lgraph = connectLayers(lgraph,"addition_1_3","conv_1_2_3");
lgraph = connectLayers(lgraph,"addition_1_3","addition_2_3/in2");
lgraph = connectLayers(lgraph,"conv_2_2_3","maxpool_2_3");
lgraph = connectLayers(lgraph,"conv_2_2_3","mul_1_2_3/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_2_3","mul_1_2_3/in1");
lgraph = connectLayers(lgraph,"mul_1_2_3","addition_2_3/in1");
lgraph = connectLayers(lgraph,"addition_2_3","conv_1_3_3");
lgraph = connectLayers(lgraph,"addition_2_3","addition_3_3/in2");
lgraph = connectLayers(lgraph,"conv_2_3_3","maxpool_3_3");
lgraph = connectLayers(lgraph,"conv_2_3_3","mul_1_3_3/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_3_3","mul_1_3_3/in1");
lgraph = connectLayers(lgraph,"mul_1_3_3","addition_3_3/in1");
lgraph = connectLayers(lgraph,"addition_3_3","conv_1_4_3");
lgraph = connectLayers(lgraph,"addition_3_3","addition_4_3/in2");
lgraph = connectLayers(lgraph,"conv_2_4_3","maxpool_4_3");
lgraph = connectLayers(lgraph,"conv_2_4_3","mul_1_4_3/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_4_3","mul_1_4_3/in1");
lgraph = connectLayers(lgraph,"mul_1_4_3","addition_4_3/in1");
lgraph = connectLayers(lgraph,"addition_4_3","conv_1_5_3");
lgraph = connectLayers(lgraph,"addition_4_3","addition_5_3/in2");
lgraph = connectLayers(lgraph,"conv_2_5_3","maxpool_5_3");
lgraph = connectLayers(lgraph,"conv_2_5_3","mul_1_5_3/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_5_3","mul_1_5_3/in1");
lgraph = connectLayers(lgraph,"mul_1_5_3","addition_5_3/in1");
lgraph = connectLayers(lgraph,"addition_5_3","conv_1_6_3");
lgraph = connectLayers(lgraph,"addition_5_3","addition_6_3/in2");
lgraph = connectLayers(lgraph,"conv_2_6_3","maxpool_6_3");
lgraph = connectLayers(lgraph,"conv_2_6_3","mul_1_6_3/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_6_3","mul_1_6_3/in1");
lgraph = connectLayers(lgraph,"mul_1_6_3","addition_6_3/in1");
lgraph = connectLayers(lgraph,"addition_6_3","conv_1_7_3");
lgraph = connectLayers(lgraph,"addition_6_3","addition_7_3/in2");
lgraph = connectLayers(lgraph,"conv_2_7_3","maxpool_7_3");
lgraph = connectLayers(lgraph,"conv_2_7_3","mul_1_7_3/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_7_3","mul_1_7_3/in1");
lgraph = connectLayers(lgraph,"mul_1_7_3","addition_7_3/in1");
lgraph = connectLayers(lgraph,"addition_7_3","conv_1_8_3");
lgraph = connectLayers(lgraph,"addition_7_3","addition_8_3/in2");
lgraph = connectLayers(lgraph,"conv_2_8_3","maxpool_8_3");
lgraph = connectLayers(lgraph,"conv_2_8_3","mul_1_8_3/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_8_3","mul_1_8_3/in1");
lgraph = connectLayers(lgraph,"mul_1_8_3","addition_8_3/in1");
lgraph = connectLayers(lgraph,"addition_8_3","conv_1_9_3");
lgraph = connectLayers(lgraph,"addition_8_3","addition_9_3/in2");
lgraph = connectLayers(lgraph,"conv_2_9_3","maxpool_9_3");
lgraph = connectLayers(lgraph,"conv_2_9_3","mul_1_9_3/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_9_3","mul_1_9_3/in1");
lgraph = connectLayers(lgraph,"mul_1_9_3","addition_9_3/in1");
lgraph = connectLayers(lgraph,"addition_9_3","conv_1_10_3");
lgraph = connectLayers(lgraph,"addition_9_3","addition_10_3/in2");
lgraph = connectLayers(lgraph,"conv_2_10_3","maxpool_10_3");
lgraph = connectLayers(lgraph,"conv_2_10_3","mul_1_10_3/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_10_3","mul_1_10_3/in1");
lgraph = connectLayers(lgraph,"mul_1_10_3","addition_10_3/in1");
lgraph = connectLayers(lgraph,"addition_10_3","conv_1_11_3");
lgraph = connectLayers(lgraph,"addition_10_3","addition_11_3/in2");
lgraph = connectLayers(lgraph,"conv_2_11_3","maxpool_11_3");
lgraph = connectLayers(lgraph,"conv_2_11_3","mul_1_11_3/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_11_3","mul_1_11_3/in1");
lgraph = connectLayers(lgraph,"mul_1_11_3","addition_11_3/in1");
lgraph = connectLayers(lgraph,"addition_11_3","conv_1_12_3");
lgraph = connectLayers(lgraph,"addition_11_3","addition_12_3/in2");
lgraph = connectLayers(lgraph,"conv_2_12_3","maxpool_12_3");
lgraph = connectLayers(lgraph,"conv_2_12_3","mul_1_12_3/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_12_3","mul_1_12_3/in1");
lgraph = connectLayers(lgraph,"mul_1_12_3","addition_12_3/in1");
lgraph = connectLayers(lgraph,"addition_12_3","conv_1_13_3");
lgraph = connectLayers(lgraph,"addition_12_3","addition_13_3/in2");
lgraph = connectLayers(lgraph,"conv_2_13_3","maxpool_13_3");
lgraph = connectLayers(lgraph,"conv_2_13_3","mul_1_13_3/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_13_3","mul_1_13_3/in1");
lgraph = connectLayers(lgraph,"mul_1_13_3","addition_13_3/in1");
lgraph = connectLayers(lgraph,"addition_13_3","conv_1_14_3");
lgraph = connectLayers(lgraph,"addition_13_3","addition_14_3/in2");
lgraph = connectLayers(lgraph,"conv_2_14_3","maxpool_14_3");
lgraph = connectLayers(lgraph,"conv_2_14_3","mul_1_14_3/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_14_3","mul_1_14_3/in1");
lgraph = connectLayers(lgraph,"mul_1_14_3","addition_14_3/in1");
lgraph = connectLayers(lgraph,"addition_14_3","conv_1_15_3");
lgraph = connectLayers(lgraph,"addition_14_3","addition_15_3/in2");
lgraph = connectLayers(lgraph,"conv_2_15_3","maxpool_15_3");
lgraph = connectLayers(lgraph,"conv_2_15_3","mul_1_15_3/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_15_3","mul_1_15_3/in1");
lgraph = connectLayers(lgraph,"mul_1_15_3","addition_15_3/in1");
lgraph = connectLayers(lgraph,"addition_15_3","conv_1_16_3");
lgraph = connectLayers(lgraph,"addition_15_3","addition_16_3/in2");
lgraph = connectLayers(lgraph,"conv_2_16_3","maxpool_16_3");
lgraph = connectLayers(lgraph,"conv_2_16_3","mul_1_16_3/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_16_3","mul_1_16_3/in1");
lgraph = connectLayers(lgraph,"mul_1_16_3","addition_16_3/in1");
lgraph = connectLayers(lgraph,"addition_16_3","conv_1_17_3");
lgraph = connectLayers(lgraph,"addition_16_3","addition_17_3/in2");
lgraph = connectLayers(lgraph,"conv_2_17_3","maxpool_17_3");
lgraph = connectLayers(lgraph,"conv_2_17_3","mul_1_17_3/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_17_3","mul_1_17_3/in1");
lgraph = connectLayers(lgraph,"mul_1_17_3","addition_17_3/in1");
lgraph = connectLayers(lgraph,"addition_17_3","conv_1_18_3");
lgraph = connectLayers(lgraph,"addition_17_3","addition_18_3/in2");
lgraph = connectLayers(lgraph,"conv_2_18_3","maxpool_18_3");
lgraph = connectLayers(lgraph,"conv_2_18_3","mul_1_18_3/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_18_3","mul_1_18_3/in1");
lgraph = connectLayers(lgraph,"mul_1_18_3","addition_18_3/in1");
lgraph = connectLayers(lgraph,"addition_18_3","conv_1_19_3");
lgraph = connectLayers(lgraph,"addition_18_3","addition_19_3/in2");
lgraph = connectLayers(lgraph,"conv_2_19_3","maxpool_19_3");
lgraph = connectLayers(lgraph,"conv_2_19_3","mul_1_19_3/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_19_3","mul_1_19_3/in1");
lgraph = connectLayers(lgraph,"mul_1_19_3","addition_19_3/in1");
lgraph = connectLayers(lgraph,"addition_19_3","conv_1_20_3");
lgraph = connectLayers(lgraph,"addition_19_3","addition_20_3/in2");
lgraph = connectLayers(lgraph,"conv_2_20_3","maxpool_20_3");
lgraph = connectLayers(lgraph,"conv_2_20_3","mul_1_20_3/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_20_3","mul_1_20_3/in1");
lgraph = connectLayers(lgraph,"mul_1_20_3","addition_20_3/in1");

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_1_4")
    reluLayer("Name","relu_1_1_4")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_1_4")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_1_4")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_1_4")
    reluLayer("Name","relu_2_1_4")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_1_4")
    helperSigmoidLayer("sigmoid_1_1_4","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_1_4",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_1_4");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_2_4")
    reluLayer("Name","relu_1_2_4")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_2_4")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_2_4")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_2_4")
    reluLayer("Name","relu_2_2_4")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_2_4")
    helperSigmoidLayer("sigmoid_1_2_4","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_2_4",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_2_4");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_3_4")
    reluLayer("Name","relu_1_3_4")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_3_4")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_3_4")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_3_4")
    reluLayer("Name","relu_2_3_4")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_3_4")
    helperSigmoidLayer("sigmoid_1_3_4","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_3_4",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_3_4");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_4_4")
    reluLayer("Name","relu_1_4_4")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_4_4")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_4_4")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_4_4")
    reluLayer("Name","relu_2_4_4")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_4_4")
    helperSigmoidLayer("sigmoid_1_4_4","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_4_4",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_4_4");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_5_4")
    reluLayer("Name","relu_1_5_4")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_5_4")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_5_4")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_5_4")
    reluLayer("Name","relu_2_5_4")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_5_4")
    helperSigmoidLayer("sigmoid_1_5_4","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_5_4",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_5_4");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_6_4")
    reluLayer("Name","relu_1_6_4")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_6_4")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_6_4")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_6_4")
    reluLayer("Name","relu_2_6_4")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_6_4")
    helperSigmoidLayer("sigmoid_1_6_4","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_6_4",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_6_4");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_7_4")
    reluLayer("Name","relu_1_7_4")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_7_4")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_7_4")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_7_4")
    reluLayer("Name","relu_2_7_4")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_7_4")
    helperSigmoidLayer("sigmoid_1_7_4","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_7_4",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_7_4");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_8_4")
    reluLayer("Name","relu_1_8_4")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_8_4")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_8_4")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_8_4")
    reluLayer("Name","relu_2_8_4")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_8_4")
    helperSigmoidLayer("sigmoid_1_8_4","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_8_4",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_8_4");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_9_4")
    reluLayer("Name","relu_1_9_4")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_9_4")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_9_4")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_9_4")
    reluLayer("Name","relu_2_9_4")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_9_4")
    helperSigmoidLayer("sigmoid_1_9_4","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_9_4",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_9_4");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_10_4")
    reluLayer("Name","relu_1_10_4")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_10_4")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_10_4")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_10_4")
    reluLayer("Name","relu_2_10_4")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_10_4")
    helperSigmoidLayer("sigmoid_1_10_4","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_10_4",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_10_4");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_11_4")
    reluLayer("Name","relu_1_11_4")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_11_4")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_11_4")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_11_4")
    reluLayer("Name","relu_2_11_4")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_11_4")
    helperSigmoidLayer("sigmoid_1_11_4","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_11_4",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_11_4");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_12_4")
    reluLayer("Name","relu_1_12_4")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_12_4")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_12_4")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_12_4")
    reluLayer("Name","relu_2_12_4")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_12_4")
    helperSigmoidLayer("sigmoid_1_12_4","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_12_4",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_12_4");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_13_4")
    reluLayer("Name","relu_1_13_4")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_13_4")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_13_4")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_13_4")
    reluLayer("Name","relu_2_13_4")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_13_4")
    helperSigmoidLayer("sigmoid_1_13_4","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_13_4",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_13_4");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_14_4")
    reluLayer("Name","relu_1_14_4")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_14_4")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_14_4")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_14_4")
    reluLayer("Name","relu_2_14_4")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_14_4")
    helperSigmoidLayer("sigmoid_1_14_4","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_14_4",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_14_4");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_15_4")
    reluLayer("Name","relu_1_15_4")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_15_4")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_15_4")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_15_4")
    reluLayer("Name","relu_2_15_4")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_15_4")
    helperSigmoidLayer("sigmoid_1_15_4","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_15_4",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_15_4");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_16_4")
    reluLayer("Name","relu_1_16_4")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_16_4")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_16_4")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_16_4")
    reluLayer("Name","relu_2_16_4")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_16_4")
    helperSigmoidLayer("sigmoid_1_16_4","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_16_4",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_16_4");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_17_4")
    reluLayer("Name","relu_1_17_4")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_17_4")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_17_4")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_17_4")
    reluLayer("Name","relu_2_17_4")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_17_4")
    helperSigmoidLayer("sigmoid_1_17_4","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_17_4",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_17_4");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_18_4")
    reluLayer("Name","relu_1_18_4")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_18_4")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_18_4")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_18_4")
    reluLayer("Name","relu_2_18_4")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_18_4")
    helperSigmoidLayer("sigmoid_1_18_4","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_18_4",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_18_4");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_19_4")
    reluLayer("Name","relu_1_19_4")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_19_4")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_19_4")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_19_4")
    reluLayer("Name","relu_2_19_4")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_19_4")
    helperSigmoidLayer("sigmoid_1_19_4","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_19_4",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_19_4");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_20_4")
    reluLayer("Name","relu_1_20_4")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_20_4")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_20_4")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_20_4")
    reluLayer("Name","relu_2_20_4")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_20_4")
    helperSigmoidLayer("sigmoid_1_20_4","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_20_4",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","addition_20_4")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4")];
lgraph = addLayers(lgraph,tempLayers);
l = [additionLayer(2,'Name','sum_1_4')];
lgraph = addLayers(lgraph,l);
lgraph = connectLayers(lgraph,'sum_1_3','sum_1_4/in1');
lgraph = connectLayers(lgraph,'conv_4','sum_1_4/in2');
lgraph = connectLayers(lgraph,"sum_1_3","conv_1_1_4");
lgraph = connectLayers(lgraph,"sum_1_3","addition_1_4/in2");
lgraph = connectLayers(lgraph,"conv_2_1_4","maxpool_1_4");
lgraph = connectLayers(lgraph,"conv_2_1_4","mul_1_1_4/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_1_4","mul_1_1_4/in1");
lgraph = connectLayers(lgraph,"mul_1_1_4","addition_1_4/in1");
lgraph = connectLayers(lgraph,"addition_1_4","conv_1_2_4");
lgraph = connectLayers(lgraph,"addition_1_4","addition_2_4/in2");
lgraph = connectLayers(lgraph,"conv_2_2_4","maxpool_2_4");
lgraph = connectLayers(lgraph,"conv_2_2_4","mul_1_2_4/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_2_4","mul_1_2_4/in1");
lgraph = connectLayers(lgraph,"mul_1_2_4","addition_2_4/in1");
lgraph = connectLayers(lgraph,"addition_2_4","conv_1_3_4");
lgraph = connectLayers(lgraph,"addition_2_4","addition_3_4/in2");
lgraph = connectLayers(lgraph,"conv_2_3_4","maxpool_3_4");
lgraph = connectLayers(lgraph,"conv_2_3_4","mul_1_3_4/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_3_4","mul_1_3_4/in1");
lgraph = connectLayers(lgraph,"mul_1_3_4","addition_3_4/in1");
lgraph = connectLayers(lgraph,"addition_3_4","conv_1_4_4");
lgraph = connectLayers(lgraph,"addition_3_4","addition_4_4/in2");
lgraph = connectLayers(lgraph,"conv_2_4_4","maxpool_4_4");
lgraph = connectLayers(lgraph,"conv_2_4_4","mul_1_4_4/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_4_4","mul_1_4_4/in1");
lgraph = connectLayers(lgraph,"mul_1_4_4","addition_4_4/in1");
lgraph = connectLayers(lgraph,"addition_4_4","conv_1_5_4");
lgraph = connectLayers(lgraph,"addition_4_4","addition_5_4/in2");
lgraph = connectLayers(lgraph,"conv_2_5_4","maxpool_5_4");
lgraph = connectLayers(lgraph,"conv_2_5_4","mul_1_5_4/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_5_4","mul_1_5_4/in1");
lgraph = connectLayers(lgraph,"mul_1_5_4","addition_5_4/in1");
lgraph = connectLayers(lgraph,"addition_5_4","conv_1_6_4");
lgraph = connectLayers(lgraph,"addition_5_4","addition_6_4/in2");
lgraph = connectLayers(lgraph,"conv_2_6_4","maxpool_6_4");
lgraph = connectLayers(lgraph,"conv_2_6_4","mul_1_6_4/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_6_4","mul_1_6_4/in1");
lgraph = connectLayers(lgraph,"mul_1_6_4","addition_6_4/in1");
lgraph = connectLayers(lgraph,"addition_6_4","conv_1_7_4");
lgraph = connectLayers(lgraph,"addition_6_4","addition_7_4/in2");
lgraph = connectLayers(lgraph,"conv_2_7_4","maxpool_7_4");
lgraph = connectLayers(lgraph,"conv_2_7_4","mul_1_7_4/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_7_4","mul_1_7_4/in1");
lgraph = connectLayers(lgraph,"mul_1_7_4","addition_7_4/in1");
lgraph = connectLayers(lgraph,"addition_7_4","conv_1_8_4");
lgraph = connectLayers(lgraph,"addition_7_4","addition_8_4/in2");
lgraph = connectLayers(lgraph,"conv_2_8_4","maxpool_8_4");
lgraph = connectLayers(lgraph,"conv_2_8_4","mul_1_8_4/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_8_4","mul_1_8_4/in1");
lgraph = connectLayers(lgraph,"mul_1_8_4","addition_8_4/in1");
lgraph = connectLayers(lgraph,"addition_8_4","conv_1_9_4");
lgraph = connectLayers(lgraph,"addition_8_4","addition_9_4/in2");
lgraph = connectLayers(lgraph,"conv_2_9_4","maxpool_9_4");
lgraph = connectLayers(lgraph,"conv_2_9_4","mul_1_9_4/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_9_4","mul_1_9_4/in1");
lgraph = connectLayers(lgraph,"mul_1_9_4","addition_9_4/in1");
lgraph = connectLayers(lgraph,"addition_9_4","conv_1_10_4");
lgraph = connectLayers(lgraph,"addition_9_4","addition_10_4/in2");
lgraph = connectLayers(lgraph,"conv_2_10_4","maxpool_10_4");
lgraph = connectLayers(lgraph,"conv_2_10_4","mul_1_10_4/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_10_4","mul_1_10_4/in1");
lgraph = connectLayers(lgraph,"mul_1_10_4","addition_10_4/in1");
lgraph = connectLayers(lgraph,"addition_10_4","conv_1_11_4");
lgraph = connectLayers(lgraph,"addition_10_4","addition_11_4/in2");
lgraph = connectLayers(lgraph,"conv_2_11_4","maxpool_11_4");
lgraph = connectLayers(lgraph,"conv_2_11_4","mul_1_11_4/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_11_4","mul_1_11_4/in1");
lgraph = connectLayers(lgraph,"mul_1_11_4","addition_11_4/in1");
lgraph = connectLayers(lgraph,"addition_11_4","conv_1_12_4");
lgraph = connectLayers(lgraph,"addition_11_4","addition_12_4/in2");
lgraph = connectLayers(lgraph,"conv_2_12_4","maxpool_12_4");
lgraph = connectLayers(lgraph,"conv_2_12_4","mul_1_12_4/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_12_4","mul_1_12_4/in1");
lgraph = connectLayers(lgraph,"mul_1_12_4","addition_12_4/in1");
lgraph = connectLayers(lgraph,"addition_12_4","conv_1_13_4");
lgraph = connectLayers(lgraph,"addition_12_4","addition_13_4/in2");
lgraph = connectLayers(lgraph,"conv_2_13_4","maxpool_13_4");
lgraph = connectLayers(lgraph,"conv_2_13_4","mul_1_13_4/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_13_4","mul_1_13_4/in1");
lgraph = connectLayers(lgraph,"mul_1_13_4","addition_13_4/in1");
lgraph = connectLayers(lgraph,"addition_13_4","conv_1_14_4");
lgraph = connectLayers(lgraph,"addition_13_4","addition_14_4/in2");
lgraph = connectLayers(lgraph,"conv_2_14_4","maxpool_14_4");
lgraph = connectLayers(lgraph,"conv_2_14_4","mul_1_14_4/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_14_4","mul_1_14_4/in1");
lgraph = connectLayers(lgraph,"mul_1_14_4","addition_14_4/in1");
lgraph = connectLayers(lgraph,"addition_14_4","conv_1_15_4");
lgraph = connectLayers(lgraph,"addition_14_4","addition_15_4/in2");
lgraph = connectLayers(lgraph,"conv_2_15_4","maxpool_15_4");
lgraph = connectLayers(lgraph,"conv_2_15_4","mul_1_15_4/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_15_4","mul_1_15_4/in1");
lgraph = connectLayers(lgraph,"mul_1_15_4","addition_15_4/in1");
lgraph = connectLayers(lgraph,"addition_15_4","conv_1_16_4");
lgraph = connectLayers(lgraph,"addition_15_4","addition_16_4/in2");
lgraph = connectLayers(lgraph,"conv_2_16_4","maxpool_16_4");
lgraph = connectLayers(lgraph,"conv_2_16_4","mul_1_16_4/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_16_4","mul_1_16_4/in1");
lgraph = connectLayers(lgraph,"mul_1_16_4","addition_16_4/in1");
lgraph = connectLayers(lgraph,"addition_16_4","conv_1_17_4");
lgraph = connectLayers(lgraph,"addition_16_4","addition_17_4/in2");
lgraph = connectLayers(lgraph,"conv_2_17_4","maxpool_17_4");
lgraph = connectLayers(lgraph,"conv_2_17_4","mul_1_17_4/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_17_4","mul_1_17_4/in1");
lgraph = connectLayers(lgraph,"mul_1_17_4","addition_17_4/in1");
lgraph = connectLayers(lgraph,"addition_17_4","conv_1_18_4");
lgraph = connectLayers(lgraph,"addition_17_4","addition_18_4/in2");
lgraph = connectLayers(lgraph,"conv_2_18_4","maxpool_18_4");
lgraph = connectLayers(lgraph,"conv_2_18_4","mul_1_18_4/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_18_4","mul_1_18_4/in1");
lgraph = connectLayers(lgraph,"mul_1_18_4","addition_18_4/in1");
lgraph = connectLayers(lgraph,"addition_18_4","conv_1_19_4");
lgraph = connectLayers(lgraph,"addition_18_4","addition_19_4/in2");
lgraph = connectLayers(lgraph,"conv_2_19_4","maxpool_19_4");
lgraph = connectLayers(lgraph,"conv_2_19_4","mul_1_19_4/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_19_4","mul_1_19_4/in1");
lgraph = connectLayers(lgraph,"mul_1_19_4","addition_19_4/in1");
lgraph = connectLayers(lgraph,"addition_19_4","conv_1_20_4");
lgraph = connectLayers(lgraph,"addition_19_4","addition_20_4/in2");
lgraph = connectLayers(lgraph,"conv_2_20_4","maxpool_20_4");
lgraph = connectLayers(lgraph,"conv_2_20_4","mul_1_20_4/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_20_4","mul_1_20_4/in1");
lgraph = connectLayers(lgraph,"mul_1_20_4","addition_20_4/in1");

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_1_5")
    reluLayer("Name","relu_1_1_5")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_1_5")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_1_5")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_1_5")
    reluLayer("Name","relu_2_1_5")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_1_5")
    helperSigmoidLayer("sigmoid_1_1_5","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_1_5",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_1_5");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_2_5")
    reluLayer("Name","relu_1_2_5")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_2_5")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_2_5")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_2_5")
    reluLayer("Name","relu_2_2_5")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_2_5")
    helperSigmoidLayer("sigmoid_1_2_5","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_2_5",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_2_5");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_3_5")
    reluLayer("Name","relu_1_3_5")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_3_5")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_3_5")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_3_5")
    reluLayer("Name","relu_2_3_5")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_3_5")
    helperSigmoidLayer("sigmoid_1_3_5","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_3_5",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_3_5");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_4_5")
    reluLayer("Name","relu_1_4_5")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_4_5")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_4_5")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_4_5")
    reluLayer("Name","relu_2_4_5")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_4_5")
    helperSigmoidLayer("sigmoid_1_4_5","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_4_5",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_4_5");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_5_5")
    reluLayer("Name","relu_1_5_5")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_5_5")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_5_5")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_5_5")
    reluLayer("Name","relu_2_5_5")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_5_5")
    helperSigmoidLayer("sigmoid_1_5_5","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_5_5",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_5_5");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_6_5")
    reluLayer("Name","relu_1_6_5")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_6_5")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_6_5")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_6_5")
    reluLayer("Name","relu_2_6_5")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_6_5")
    helperSigmoidLayer("sigmoid_1_6_5","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_6_5",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_6_5");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_7_5")
    reluLayer("Name","relu_1_7_5")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_7_5")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_7_5")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_7_5")
    reluLayer("Name","relu_2_7_5")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_7_5")
    helperSigmoidLayer("sigmoid_1_7_5","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_7_5",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_7_5");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_8_5")
    reluLayer("Name","relu_1_8_5")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_8_5")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_8_5")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_8_5")
    reluLayer("Name","relu_2_8_5")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_8_5")
    helperSigmoidLayer("sigmoid_1_8_5","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_8_5",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_8_5");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_9_5")
    reluLayer("Name","relu_1_9_5")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_9_5")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_9_5")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_9_5")
    reluLayer("Name","relu_2_9_5")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_9_5")
    helperSigmoidLayer("sigmoid_1_9_5","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_9_5",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_9_5");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_10_5")
    reluLayer("Name","relu_1_10_5")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_10_5")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_10_5")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_10_5")
    reluLayer("Name","relu_2_10_5")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_10_5")
    helperSigmoidLayer("sigmoid_1_10_5","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_10_5",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_10_5");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_11_5")
    reluLayer("Name","relu_1_11_5")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_11_5")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_11_5")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_11_5")
    reluLayer("Name","relu_2_11_5")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_11_5")
    helperSigmoidLayer("sigmoid_1_11_5","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_11_5",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_11_5");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_12_5")
    reluLayer("Name","relu_1_12_5")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_12_5")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_12_5")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_12_5")
    reluLayer("Name","relu_2_12_5")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_12_5")
    helperSigmoidLayer("sigmoid_1_12_5","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_12_5",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_12_5");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_13_5")
    reluLayer("Name","relu_1_13_5")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_13_5")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_13_5")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_13_5")
    reluLayer("Name","relu_2_13_5")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_13_5")
    helperSigmoidLayer("sigmoid_1_13_5","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_13_5",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_13_5");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_14_5")
    reluLayer("Name","relu_1_14_5")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_14_5")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_14_5")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_14_5")
    reluLayer("Name","relu_2_14_5")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_14_5")
    helperSigmoidLayer("sigmoid_1_14_5","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_14_5",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_14_5");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_15_5")
    reluLayer("Name","relu_1_15_5")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_15_5")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_15_5")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_15_5")
    reluLayer("Name","relu_2_15_5")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_15_5")
    helperSigmoidLayer("sigmoid_1_15_5","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_15_5",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_15_5");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_16_5")
    reluLayer("Name","relu_1_16_5")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_16_5")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_16_5")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_16_5")
    reluLayer("Name","relu_2_16_5")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_16_5")
    helperSigmoidLayer("sigmoid_1_16_5","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_16_5",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_16_5");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_17_5")
    reluLayer("Name","relu_1_17_5")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_17_5")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_17_5")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_17_5")
    reluLayer("Name","relu_2_17_5")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_17_5")
    helperSigmoidLayer("sigmoid_1_17_5","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_17_5",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_17_5");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_18_5")
    reluLayer("Name","relu_1_18_5")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_18_5")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_18_5")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_18_5")
    reluLayer("Name","relu_2_18_5")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_18_5")
    helperSigmoidLayer("sigmoid_1_18_5","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_18_5",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_18_5");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_19_5")
    reluLayer("Name","relu_1_19_5")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_19_5")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_19_5")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_19_5")
    reluLayer("Name","relu_2_19_5")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_19_5")
    helperSigmoidLayer("sigmoid_1_19_5","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_19_5",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_19_5");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_20_5")
    reluLayer("Name","relu_1_20_5")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_20_5")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_20_5")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_20_5")
    reluLayer("Name","relu_2_20_5")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_20_5")
    helperSigmoidLayer("sigmoid_1_20_5","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_20_5",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","addition_20_5")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_5")];
lgraph = addLayers(lgraph,tempLayers);
l = [additionLayer(2,'Name','sum_1_5')];
lgraph = addLayers(lgraph,l);
lgraph = connectLayers(lgraph,'sum_1_4','sum_1_5/in1');
lgraph = connectLayers(lgraph,'conv_5','sum_1_5/in2');
lgraph = connectLayers(lgraph,"sum_1_4","conv_1_1_5");
lgraph = connectLayers(lgraph,"sum_1_4","addition_1_5/in2");
lgraph = connectLayers(lgraph,"conv_2_1_5","maxpool_1_5");
lgraph = connectLayers(lgraph,"conv_2_1_5","mul_1_1_5/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_1_5","mul_1_1_5/in1");
lgraph = connectLayers(lgraph,"mul_1_1_5","addition_1_5/in1");
lgraph = connectLayers(lgraph,"addition_1_5","conv_1_2_5");
lgraph = connectLayers(lgraph,"addition_1_5","addition_2_5/in2");
lgraph = connectLayers(lgraph,"conv_2_2_5","maxpool_2_5");
lgraph = connectLayers(lgraph,"conv_2_2_5","mul_1_2_5/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_2_5","mul_1_2_5/in1");
lgraph = connectLayers(lgraph,"mul_1_2_5","addition_2_5/in1");
lgraph = connectLayers(lgraph,"addition_2_5","conv_1_3_5");
lgraph = connectLayers(lgraph,"addition_2_5","addition_3_5/in2");
lgraph = connectLayers(lgraph,"conv_2_3_5","maxpool_3_5");
lgraph = connectLayers(lgraph,"conv_2_3_5","mul_1_3_5/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_3_5","mul_1_3_5/in1");
lgraph = connectLayers(lgraph,"mul_1_3_5","addition_3_5/in1");
lgraph = connectLayers(lgraph,"addition_3_5","conv_1_4_5");
lgraph = connectLayers(lgraph,"addition_3_5","addition_4_5/in2");
lgraph = connectLayers(lgraph,"conv_2_4_5","maxpool_4_5");
lgraph = connectLayers(lgraph,"conv_2_4_5","mul_1_4_5/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_4_5","mul_1_4_5/in1");
lgraph = connectLayers(lgraph,"mul_1_4_5","addition_4_5/in1");
lgraph = connectLayers(lgraph,"addition_4_5","conv_1_5_5");
lgraph = connectLayers(lgraph,"addition_4_5","addition_5_5/in2");
lgraph = connectLayers(lgraph,"conv_2_5_5","maxpool_5_5");
lgraph = connectLayers(lgraph,"conv_2_5_5","mul_1_5_5/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_5_5","mul_1_5_5/in1");
lgraph = connectLayers(lgraph,"mul_1_5_5","addition_5_5/in1");
lgraph = connectLayers(lgraph,"addition_5_5","conv_1_6_5");
lgraph = connectLayers(lgraph,"addition_5_5","addition_6_5/in2");
lgraph = connectLayers(lgraph,"conv_2_6_5","maxpool_6_5");
lgraph = connectLayers(lgraph,"conv_2_6_5","mul_1_6_5/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_6_5","mul_1_6_5/in1");
lgraph = connectLayers(lgraph,"mul_1_6_5","addition_6_5/in1");
lgraph = connectLayers(lgraph,"addition_6_5","conv_1_7_5");
lgraph = connectLayers(lgraph,"addition_6_5","addition_7_5/in2");
lgraph = connectLayers(lgraph,"conv_2_7_5","maxpool_7_5");
lgraph = connectLayers(lgraph,"conv_2_7_5","mul_1_7_5/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_7_5","mul_1_7_5/in1");
lgraph = connectLayers(lgraph,"mul_1_7_5","addition_7_5/in1");
lgraph = connectLayers(lgraph,"addition_7_5","conv_1_8_5");
lgraph = connectLayers(lgraph,"addition_7_5","addition_8_5/in2");
lgraph = connectLayers(lgraph,"conv_2_8_5","maxpool_8_5");
lgraph = connectLayers(lgraph,"conv_2_8_5","mul_1_8_5/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_8_5","mul_1_8_5/in1");
lgraph = connectLayers(lgraph,"mul_1_8_5","addition_8_5/in1");
lgraph = connectLayers(lgraph,"addition_8_5","conv_1_9_5");
lgraph = connectLayers(lgraph,"addition_8_5","addition_9_5/in2");
lgraph = connectLayers(lgraph,"conv_2_9_5","maxpool_9_5");
lgraph = connectLayers(lgraph,"conv_2_9_5","mul_1_9_5/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_9_5","mul_1_9_5/in1");
lgraph = connectLayers(lgraph,"mul_1_9_5","addition_9_5/in1");
lgraph = connectLayers(lgraph,"addition_9_5","conv_1_10_5");
lgraph = connectLayers(lgraph,"addition_9_5","addition_10_5/in2");
lgraph = connectLayers(lgraph,"conv_2_10_5","maxpool_10_5");
lgraph = connectLayers(lgraph,"conv_2_10_5","mul_1_10_5/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_10_5","mul_1_10_5/in1");
lgraph = connectLayers(lgraph,"mul_1_10_5","addition_10_5/in1");
lgraph = connectLayers(lgraph,"addition_10_5","conv_1_11_5");
lgraph = connectLayers(lgraph,"addition_10_5","addition_11_5/in2");
lgraph = connectLayers(lgraph,"conv_2_11_5","maxpool_11_5");
lgraph = connectLayers(lgraph,"conv_2_11_5","mul_1_11_5/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_11_5","mul_1_11_5/in1");
lgraph = connectLayers(lgraph,"mul_1_11_5","addition_11_5/in1");
lgraph = connectLayers(lgraph,"addition_11_5","conv_1_12_5");
lgraph = connectLayers(lgraph,"addition_11_5","addition_12_5/in2");
lgraph = connectLayers(lgraph,"conv_2_12_5","maxpool_12_5");
lgraph = connectLayers(lgraph,"conv_2_12_5","mul_1_12_5/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_12_5","mul_1_12_5/in1");
lgraph = connectLayers(lgraph,"mul_1_12_5","addition_12_5/in1");
lgraph = connectLayers(lgraph,"addition_12_5","conv_1_13_5");
lgraph = connectLayers(lgraph,"addition_12_5","addition_13_5/in2");
lgraph = connectLayers(lgraph,"conv_2_13_5","maxpool_13_5");
lgraph = connectLayers(lgraph,"conv_2_13_5","mul_1_13_5/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_13_5","mul_1_13_5/in1");
lgraph = connectLayers(lgraph,"mul_1_13_5","addition_13_5/in1");
lgraph = connectLayers(lgraph,"addition_13_5","conv_1_14_5");
lgraph = connectLayers(lgraph,"addition_13_5","addition_14_5/in2");
lgraph = connectLayers(lgraph,"conv_2_14_5","maxpool_14_5");
lgraph = connectLayers(lgraph,"conv_2_14_5","mul_1_14_5/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_14_5","mul_1_14_5/in1");
lgraph = connectLayers(lgraph,"mul_1_14_5","addition_14_5/in1");
lgraph = connectLayers(lgraph,"addition_14_5","conv_1_15_5");
lgraph = connectLayers(lgraph,"addition_14_5","addition_15_5/in2");
lgraph = connectLayers(lgraph,"conv_2_15_5","maxpool_15_5");
lgraph = connectLayers(lgraph,"conv_2_15_5","mul_1_15_5/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_15_5","mul_1_15_5/in1");
lgraph = connectLayers(lgraph,"mul_1_15_5","addition_15_5/in1");
lgraph = connectLayers(lgraph,"addition_15_5","conv_1_16_5");
lgraph = connectLayers(lgraph,"addition_15_5","addition_16_5/in2");
lgraph = connectLayers(lgraph,"conv_2_16_5","maxpool_16_5");
lgraph = connectLayers(lgraph,"conv_2_16_5","mul_1_16_5/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_16_5","mul_1_16_5/in1");
lgraph = connectLayers(lgraph,"mul_1_16_5","addition_16_5/in1");
lgraph = connectLayers(lgraph,"addition_16_5","conv_1_17_5");
lgraph = connectLayers(lgraph,"addition_16_5","addition_17_5/in2");
lgraph = connectLayers(lgraph,"conv_2_17_5","maxpool_17_5");
lgraph = connectLayers(lgraph,"conv_2_17_5","mul_1_17_5/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_17_5","mul_1_17_5/in1");
lgraph = connectLayers(lgraph,"mul_1_17_5","addition_17_5/in1");
lgraph = connectLayers(lgraph,"addition_17_5","conv_1_18_5");
lgraph = connectLayers(lgraph,"addition_17_5","addition_18_5/in2");
lgraph = connectLayers(lgraph,"conv_2_18_5","maxpool_18_5");
lgraph = connectLayers(lgraph,"conv_2_18_5","mul_1_18_5/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_18_5","mul_1_18_5/in1");
lgraph = connectLayers(lgraph,"mul_1_18_5","addition_18_5/in1");
lgraph = connectLayers(lgraph,"addition_18_5","conv_1_19_5");
lgraph = connectLayers(lgraph,"addition_18_5","addition_19_5/in2");
lgraph = connectLayers(lgraph,"conv_2_19_5","maxpool_19_5");
lgraph = connectLayers(lgraph,"conv_2_19_5","mul_1_19_5/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_19_5","mul_1_19_5/in1");
lgraph = connectLayers(lgraph,"mul_1_19_5","addition_19_5/in1");
lgraph = connectLayers(lgraph,"addition_19_5","conv_1_20_5");
lgraph = connectLayers(lgraph,"addition_19_5","addition_20_5/in2");
lgraph = connectLayers(lgraph,"conv_2_20_5","maxpool_20_5");
lgraph = connectLayers(lgraph,"conv_2_20_5","mul_1_20_5/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_20_5","mul_1_20_5/in1");
lgraph = connectLayers(lgraph,"mul_1_20_5","addition_20_5/in1");

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_1_6")
    reluLayer("Name","relu_1_1_6")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_1_6")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_1_6")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_1_6")
    reluLayer("Name","relu_2_1_6")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_1_6")
    helperSigmoidLayer("sigmoid_1_1_6","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_1_6",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_1_6");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_2_6")
    reluLayer("Name","relu_1_2_6")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_2_6")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_2_6")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_2_6")
    reluLayer("Name","relu_2_2_6")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_2_6")
    helperSigmoidLayer("sigmoid_1_2_6","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_2_6",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_2_6");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_3_6")
    reluLayer("Name","relu_1_3_6")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_3_6")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_3_6")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_3_6")
    reluLayer("Name","relu_2_3_6")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_3_6")
    helperSigmoidLayer("sigmoid_1_3_6","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_3_6",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_3_6");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_4_6")
    reluLayer("Name","relu_1_4_6")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_4_6")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_4_6")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_4_6")
    reluLayer("Name","relu_2_4_6")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_4_6")
    helperSigmoidLayer("sigmoid_1_4_6","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_4_6",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_4_6");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_5_6")
    reluLayer("Name","relu_1_5_6")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_5_6")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_5_6")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_5_6")
    reluLayer("Name","relu_2_5_6")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_5_6")
    helperSigmoidLayer("sigmoid_1_5_6","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_5_6",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_5_6");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_6_6")
    reluLayer("Name","relu_1_6_6")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_6_6")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_6_6")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_6_6")
    reluLayer("Name","relu_2_6_6")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_6_6")
    helperSigmoidLayer("sigmoid_1_6_6","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_6_6",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_6_6");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_7_6")
    reluLayer("Name","relu_1_7_6")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_7_6")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_7_6")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_7_6")
    reluLayer("Name","relu_2_7_6")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_7_6")
    helperSigmoidLayer("sigmoid_1_7_6","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_7_6",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_7_6");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_8_6")
    reluLayer("Name","relu_1_8_6")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_8_6")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_8_6")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_8_6")
    reluLayer("Name","relu_2_8_6")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_8_6")
    helperSigmoidLayer("sigmoid_1_8_6","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_8_6",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_8_6");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_9_6")
    reluLayer("Name","relu_1_9_6")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_9_6")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_9_6")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_9_6")
    reluLayer("Name","relu_2_9_6")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_9_6")
    helperSigmoidLayer("sigmoid_1_9_6","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_9_6",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_9_6");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_10_6")
    reluLayer("Name","relu_1_10_6")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_10_6")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_10_6")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_10_6")
    reluLayer("Name","relu_2_10_6")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_10_6")
    helperSigmoidLayer("sigmoid_1_10_6","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_10_6",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_10_6");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_11_6")
    reluLayer("Name","relu_1_11_6")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_11_6")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_11_6")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_11_6")
    reluLayer("Name","relu_2_11_6")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_11_6")
    helperSigmoidLayer("sigmoid_1_11_6","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_11_6",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_11_6");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_12_6")
    reluLayer("Name","relu_1_12_6")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_12_6")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_12_6")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_12_6")
    reluLayer("Name","relu_2_12_6")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_12_6")
    helperSigmoidLayer("sigmoid_1_12_6","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_12_6",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_12_6");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_13_6")
    reluLayer("Name","relu_1_13_6")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_13_6")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_13_6")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_13_6")
    reluLayer("Name","relu_2_13_6")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_13_6")
    helperSigmoidLayer("sigmoid_1_13_6","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_13_6",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_13_6");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_14_6")
    reluLayer("Name","relu_1_14_6")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_14_6")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_14_6")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_14_6")
    reluLayer("Name","relu_2_14_6")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_14_6")
    helperSigmoidLayer("sigmoid_1_14_6","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_14_6",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_14_6");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_15_6")
    reluLayer("Name","relu_1_15_6")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_15_6")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_15_6")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_15_6")
    reluLayer("Name","relu_2_15_6")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_15_6")
    helperSigmoidLayer("sigmoid_1_15_6","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_15_6",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_15_6");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_16_6")
    reluLayer("Name","relu_1_16_6")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_16_6")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_16_6")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_16_6")
    reluLayer("Name","relu_2_16_6")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_16_6")
    helperSigmoidLayer("sigmoid_1_16_6","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_16_6",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_16_6");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_17_6")
    reluLayer("Name","relu_1_17_6")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_17_6")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_17_6")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_17_6")
    reluLayer("Name","relu_2_17_6")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_17_6")
    helperSigmoidLayer("sigmoid_1_17_6","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_17_6",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_17_6");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_18_6")
    reluLayer("Name","relu_1_18_6")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_18_6")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_18_6")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_18_6")
    reluLayer("Name","relu_2_18_6")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_18_6")
    helperSigmoidLayer("sigmoid_1_18_6","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_18_6",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_18_6");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_19_6")
    reluLayer("Name","relu_1_19_6")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_19_6")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_19_6")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_19_6")
    reluLayer("Name","relu_2_19_6")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_19_6")
    helperSigmoidLayer("sigmoid_1_19_6","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_19_6",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_19_6");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_20_6")
    reluLayer("Name","relu_1_20_6")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_20_6")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_20_6")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_20_6")
    reluLayer("Name","relu_2_20_6")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_20_6")
    helperSigmoidLayer("sigmoid_1_20_6","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_20_6",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","addition_20_6")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_6")];
lgraph = addLayers(lgraph,tempLayers);
l = [additionLayer(2,'Name','sum_1_6')];
lgraph = addLayers(lgraph,l);
lgraph = connectLayers(lgraph,'sum_1_5','sum_1_6/in1');
lgraph = connectLayers(lgraph,'conv_6','sum_1_6/in2');
lgraph = connectLayers(lgraph,"sum_1_5","conv_1_1_6");
lgraph = connectLayers(lgraph,"sum_1_5","addition_1_6/in2");
lgraph = connectLayers(lgraph,"conv_2_1_6","maxpool_1_6");
lgraph = connectLayers(lgraph,"conv_2_1_6","mul_1_1_6/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_1_6","mul_1_1_6/in1");
lgraph = connectLayers(lgraph,"mul_1_1_6","addition_1_6/in1");
lgraph = connectLayers(lgraph,"addition_1_6","conv_1_2_6");
lgraph = connectLayers(lgraph,"addition_1_6","addition_2_6/in2");
lgraph = connectLayers(lgraph,"conv_2_2_6","maxpool_2_6");
lgraph = connectLayers(lgraph,"conv_2_2_6","mul_1_2_6/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_2_6","mul_1_2_6/in1");
lgraph = connectLayers(lgraph,"mul_1_2_6","addition_2_6/in1");
lgraph = connectLayers(lgraph,"addition_2_6","conv_1_3_6");
lgraph = connectLayers(lgraph,"addition_2_6","addition_3_6/in2");
lgraph = connectLayers(lgraph,"conv_2_3_6","maxpool_3_6");
lgraph = connectLayers(lgraph,"conv_2_3_6","mul_1_3_6/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_3_6","mul_1_3_6/in1");
lgraph = connectLayers(lgraph,"mul_1_3_6","addition_3_6/in1");
lgraph = connectLayers(lgraph,"addition_3_6","conv_1_4_6");
lgraph = connectLayers(lgraph,"addition_3_6","addition_4_6/in2");
lgraph = connectLayers(lgraph,"conv_2_4_6","maxpool_4_6");
lgraph = connectLayers(lgraph,"conv_2_4_6","mul_1_4_6/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_4_6","mul_1_4_6/in1");
lgraph = connectLayers(lgraph,"mul_1_4_6","addition_4_6/in1");
lgraph = connectLayers(lgraph,"addition_4_6","conv_1_5_6");
lgraph = connectLayers(lgraph,"addition_4_6","addition_5_6/in2");
lgraph = connectLayers(lgraph,"conv_2_5_6","maxpool_5_6");
lgraph = connectLayers(lgraph,"conv_2_5_6","mul_1_5_6/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_5_6","mul_1_5_6/in1");
lgraph = connectLayers(lgraph,"mul_1_5_6","addition_5_6/in1");
lgraph = connectLayers(lgraph,"addition_5_6","conv_1_6_6");
lgraph = connectLayers(lgraph,"addition_5_6","addition_6_6/in2");
lgraph = connectLayers(lgraph,"conv_2_6_6","maxpool_6_6");
lgraph = connectLayers(lgraph,"conv_2_6_6","mul_1_6_6/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_6_6","mul_1_6_6/in1");
lgraph = connectLayers(lgraph,"mul_1_6_6","addition_6_6/in1");
lgraph = connectLayers(lgraph,"addition_6_6","conv_1_7_6");
lgraph = connectLayers(lgraph,"addition_6_6","addition_7_6/in2");
lgraph = connectLayers(lgraph,"conv_2_7_6","maxpool_7_6");
lgraph = connectLayers(lgraph,"conv_2_7_6","mul_1_7_6/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_7_6","mul_1_7_6/in1");
lgraph = connectLayers(lgraph,"mul_1_7_6","addition_7_6/in1");
lgraph = connectLayers(lgraph,"addition_7_6","conv_1_8_6");
lgraph = connectLayers(lgraph,"addition_7_6","addition_8_6/in2");
lgraph = connectLayers(lgraph,"conv_2_8_6","maxpool_8_6");
lgraph = connectLayers(lgraph,"conv_2_8_6","mul_1_8_6/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_8_6","mul_1_8_6/in1");
lgraph = connectLayers(lgraph,"mul_1_8_6","addition_8_6/in1");
lgraph = connectLayers(lgraph,"addition_8_6","conv_1_9_6");
lgraph = connectLayers(lgraph,"addition_8_6","addition_9_6/in2");
lgraph = connectLayers(lgraph,"conv_2_9_6","maxpool_9_6");
lgraph = connectLayers(lgraph,"conv_2_9_6","mul_1_9_6/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_9_6","mul_1_9_6/in1");
lgraph = connectLayers(lgraph,"mul_1_9_6","addition_9_6/in1");
lgraph = connectLayers(lgraph,"addition_9_6","conv_1_10_6");
lgraph = connectLayers(lgraph,"addition_9_6","addition_10_6/in2");
lgraph = connectLayers(lgraph,"conv_2_10_6","maxpool_10_6");
lgraph = connectLayers(lgraph,"conv_2_10_6","mul_1_10_6/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_10_6","mul_1_10_6/in1");
lgraph = connectLayers(lgraph,"mul_1_10_6","addition_10_6/in1");
lgraph = connectLayers(lgraph,"addition_10_6","conv_1_11_6");
lgraph = connectLayers(lgraph,"addition_10_6","addition_11_6/in2");
lgraph = connectLayers(lgraph,"conv_2_11_6","maxpool_11_6");
lgraph = connectLayers(lgraph,"conv_2_11_6","mul_1_11_6/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_11_6","mul_1_11_6/in1");
lgraph = connectLayers(lgraph,"mul_1_11_6","addition_11_6/in1");
lgraph = connectLayers(lgraph,"addition_11_6","conv_1_12_6");
lgraph = connectLayers(lgraph,"addition_11_6","addition_12_6/in2");
lgraph = connectLayers(lgraph,"conv_2_12_6","maxpool_12_6");
lgraph = connectLayers(lgraph,"conv_2_12_6","mul_1_12_6/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_12_6","mul_1_12_6/in1");
lgraph = connectLayers(lgraph,"mul_1_12_6","addition_12_6/in1");
lgraph = connectLayers(lgraph,"addition_12_6","conv_1_13_6");
lgraph = connectLayers(lgraph,"addition_12_6","addition_13_6/in2");
lgraph = connectLayers(lgraph,"conv_2_13_6","maxpool_13_6");
lgraph = connectLayers(lgraph,"conv_2_13_6","mul_1_13_6/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_13_6","mul_1_13_6/in1");
lgraph = connectLayers(lgraph,"mul_1_13_6","addition_13_6/in1");
lgraph = connectLayers(lgraph,"addition_13_6","conv_1_14_6");
lgraph = connectLayers(lgraph,"addition_13_6","addition_14_6/in2");
lgraph = connectLayers(lgraph,"conv_2_14_6","maxpool_14_6");
lgraph = connectLayers(lgraph,"conv_2_14_6","mul_1_14_6/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_14_6","mul_1_14_6/in1");
lgraph = connectLayers(lgraph,"mul_1_14_6","addition_14_6/in1");
lgraph = connectLayers(lgraph,"addition_14_6","conv_1_15_6");
lgraph = connectLayers(lgraph,"addition_14_6","addition_15_6/in2");
lgraph = connectLayers(lgraph,"conv_2_15_6","maxpool_15_6");
lgraph = connectLayers(lgraph,"conv_2_15_6","mul_1_15_6/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_15_6","mul_1_15_6/in1");
lgraph = connectLayers(lgraph,"mul_1_15_6","addition_15_6/in1");
lgraph = connectLayers(lgraph,"addition_15_6","conv_1_16_6");
lgraph = connectLayers(lgraph,"addition_15_6","addition_16_6/in2");
lgraph = connectLayers(lgraph,"conv_2_16_6","maxpool_16_6");
lgraph = connectLayers(lgraph,"conv_2_16_6","mul_1_16_6/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_16_6","mul_1_16_6/in1");
lgraph = connectLayers(lgraph,"mul_1_16_6","addition_16_6/in1");
lgraph = connectLayers(lgraph,"addition_16_6","conv_1_17_6");
lgraph = connectLayers(lgraph,"addition_16_6","addition_17_6/in2");
lgraph = connectLayers(lgraph,"conv_2_17_6","maxpool_17_6");
lgraph = connectLayers(lgraph,"conv_2_17_6","mul_1_17_6/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_17_6","mul_1_17_6/in1");
lgraph = connectLayers(lgraph,"mul_1_17_6","addition_17_6/in1");
lgraph = connectLayers(lgraph,"addition_17_6","conv_1_18_6");
lgraph = connectLayers(lgraph,"addition_17_6","addition_18_6/in2");
lgraph = connectLayers(lgraph,"conv_2_18_6","maxpool_18_6");
lgraph = connectLayers(lgraph,"conv_2_18_6","mul_1_18_6/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_18_6","mul_1_18_6/in1");
lgraph = connectLayers(lgraph,"mul_1_18_6","addition_18_6/in1");
lgraph = connectLayers(lgraph,"addition_18_6","conv_1_19_6");
lgraph = connectLayers(lgraph,"addition_18_6","addition_19_6/in2");
lgraph = connectLayers(lgraph,"conv_2_19_6","maxpool_19_6");
lgraph = connectLayers(lgraph,"conv_2_19_6","mul_1_19_6/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_19_6","mul_1_19_6/in1");
lgraph = connectLayers(lgraph,"mul_1_19_6","addition_19_6/in1");
lgraph = connectLayers(lgraph,"addition_19_6","conv_1_20_6");
lgraph = connectLayers(lgraph,"addition_19_6","addition_20_6/in2");
lgraph = connectLayers(lgraph,"conv_2_20_6","maxpool_20_6");
lgraph = connectLayers(lgraph,"conv_2_20_6","mul_1_20_6/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_20_6","mul_1_20_6/in1");
lgraph = connectLayers(lgraph,"mul_1_20_6","addition_20_6/in1");

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_1_7")
    reluLayer("Name","relu_1_1_7")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_1_7")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_1_7")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_1_7")
    reluLayer("Name","relu_2_1_7")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_1_7")
    helperSigmoidLayer("sigmoid_1_1_7","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_1_7",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_1_7");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_2_7")
    reluLayer("Name","relu_1_2_7")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_2_7")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_2_7")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_2_7")
    reluLayer("Name","relu_2_2_7")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_2_7")
    helperSigmoidLayer("sigmoid_1_2_7","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_2_7",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_2_7");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_3_7")
    reluLayer("Name","relu_1_3_7")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_3_7")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_3_7")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_3_7")
    reluLayer("Name","relu_2_3_7")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_3_7")
    helperSigmoidLayer("sigmoid_1_3_7","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_3_7",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_3_7");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_4_7")
    reluLayer("Name","relu_1_4_7")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_4_7")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_4_7")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_4_7")
    reluLayer("Name","relu_2_4_7")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_4_7")
    helperSigmoidLayer("sigmoid_1_4_7","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_4_7",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_4_7");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_5_7")
    reluLayer("Name","relu_1_5_7")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_5_7")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_5_7")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_5_7")
    reluLayer("Name","relu_2_5_7")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_5_7")
    helperSigmoidLayer("sigmoid_1_5_7","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_5_7",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_5_7");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_6_7")
    reluLayer("Name","relu_1_6_7")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_6_7")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_6_7")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_6_7")
    reluLayer("Name","relu_2_6_7")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_6_7")
    helperSigmoidLayer("sigmoid_1_6_7","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_6_7",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_6_7");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_7_7")
    reluLayer("Name","relu_1_7_7")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_7_7")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_7_7")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_7_7")
    reluLayer("Name","relu_2_7_7")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_7_7")
    helperSigmoidLayer("sigmoid_1_7_7","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_7_7",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_7_7");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_8_7")
    reluLayer("Name","relu_1_8_7")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_8_7")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_8_7")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_8_7")
    reluLayer("Name","relu_2_8_7")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_8_7")
    helperSigmoidLayer("sigmoid_1_8_7","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_8_7",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_8_7");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_9_7")
    reluLayer("Name","relu_1_9_7")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_9_7")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_9_7")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_9_7")
    reluLayer("Name","relu_2_9_7")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_9_7")
    helperSigmoidLayer("sigmoid_1_9_7","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_9_7",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_9_7");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_10_7")
    reluLayer("Name","relu_1_10_7")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_10_7")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_10_7")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_10_7")
    reluLayer("Name","relu_2_10_7")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_10_7")
    helperSigmoidLayer("sigmoid_1_10_7","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_10_7",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_10_7");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_11_7")
    reluLayer("Name","relu_1_11_7")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_11_7")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_11_7")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_11_7")
    reluLayer("Name","relu_2_11_7")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_11_7")
    helperSigmoidLayer("sigmoid_1_11_7","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_11_7",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_11_7");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_12_7")
    reluLayer("Name","relu_1_12_7")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_12_7")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_12_7")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_12_7")
    reluLayer("Name","relu_2_12_7")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_12_7")
    helperSigmoidLayer("sigmoid_1_12_7","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_12_7",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_12_7");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_13_7")
    reluLayer("Name","relu_1_13_7")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_13_7")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_13_7")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_13_7")
    reluLayer("Name","relu_2_13_7")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_13_7")
    helperSigmoidLayer("sigmoid_1_13_7","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_13_7",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_13_7");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_14_7")
    reluLayer("Name","relu_1_14_7")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_14_7")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_14_7")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_14_7")
    reluLayer("Name","relu_2_14_7")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_14_7")
    helperSigmoidLayer("sigmoid_1_14_7","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_14_7",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_14_7");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_15_7")
    reluLayer("Name","relu_1_15_7")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_15_7")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_15_7")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_15_7")
    reluLayer("Name","relu_2_15_7")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_15_7")
    helperSigmoidLayer("sigmoid_1_15_7","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_15_7",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_15_7");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_16_7")
    reluLayer("Name","relu_1_16_7")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_16_7")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_16_7")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_16_7")
    reluLayer("Name","relu_2_16_7")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_16_7")
    helperSigmoidLayer("sigmoid_1_16_7","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_16_7",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_16_7");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_17_7")
    reluLayer("Name","relu_1_17_7")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_17_7")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_17_7")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_17_7")
    reluLayer("Name","relu_2_17_7")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_17_7")
    helperSigmoidLayer("sigmoid_1_17_7","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_17_7",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_17_7");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_18_7")
    reluLayer("Name","relu_1_18_7")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_18_7")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_18_7")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_18_7")
    reluLayer("Name","relu_2_18_7")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_18_7")
    helperSigmoidLayer("sigmoid_1_18_7","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_18_7",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_18_7");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_19_7")
    reluLayer("Name","relu_1_19_7")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_19_7")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_19_7")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_19_7")
    reluLayer("Name","relu_2_19_7")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_19_7")
    helperSigmoidLayer("sigmoid_1_19_7","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_19_7",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_19_7");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_20_7")
    reluLayer("Name","relu_1_20_7")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_20_7")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_20_7")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_20_7")
    reluLayer("Name","relu_2_20_7")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_20_7")
    helperSigmoidLayer("sigmoid_1_20_7","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_20_7",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","addition_20_7")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_7")];
lgraph = addLayers(lgraph,tempLayers);
l = [additionLayer(2,'Name','sum_1_7')];
lgraph = addLayers(lgraph,l);
lgraph = connectLayers(lgraph,'sum_1_6','sum_1_7/in1');
lgraph = connectLayers(lgraph,'conv_7','sum_1_7/in2');
lgraph = connectLayers(lgraph,"sum_1_6","conv_1_1_7");
lgraph = connectLayers(lgraph,"sum_1_6","addition_1_7/in2");
lgraph = connectLayers(lgraph,"conv_2_1_7","maxpool_1_7");
lgraph = connectLayers(lgraph,"conv_2_1_7","mul_1_1_7/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_1_7","mul_1_1_7/in1");
lgraph = connectLayers(lgraph,"mul_1_1_7","addition_1_7/in1");
lgraph = connectLayers(lgraph,"addition_1_7","conv_1_2_7");
lgraph = connectLayers(lgraph,"addition_1_7","addition_2_7/in2");
lgraph = connectLayers(lgraph,"conv_2_2_7","maxpool_2_7");
lgraph = connectLayers(lgraph,"conv_2_2_7","mul_1_2_7/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_2_7","mul_1_2_7/in1");
lgraph = connectLayers(lgraph,"mul_1_2_7","addition_2_7/in1");
lgraph = connectLayers(lgraph,"addition_2_7","conv_1_3_7");
lgraph = connectLayers(lgraph,"addition_2_7","addition_3_7/in2");
lgraph = connectLayers(lgraph,"conv_2_3_7","maxpool_3_7");
lgraph = connectLayers(lgraph,"conv_2_3_7","mul_1_3_7/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_3_7","mul_1_3_7/in1");
lgraph = connectLayers(lgraph,"mul_1_3_7","addition_3_7/in1");
lgraph = connectLayers(lgraph,"addition_3_7","conv_1_4_7");
lgraph = connectLayers(lgraph,"addition_3_7","addition_4_7/in2");
lgraph = connectLayers(lgraph,"conv_2_4_7","maxpool_4_7");
lgraph = connectLayers(lgraph,"conv_2_4_7","mul_1_4_7/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_4_7","mul_1_4_7/in1");
lgraph = connectLayers(lgraph,"mul_1_4_7","addition_4_7/in1");
lgraph = connectLayers(lgraph,"addition_4_7","conv_1_5_7");
lgraph = connectLayers(lgraph,"addition_4_7","addition_5_7/in2");
lgraph = connectLayers(lgraph,"conv_2_5_7","maxpool_5_7");
lgraph = connectLayers(lgraph,"conv_2_5_7","mul_1_5_7/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_5_7","mul_1_5_7/in1");
lgraph = connectLayers(lgraph,"mul_1_5_7","addition_5_7/in1");
lgraph = connectLayers(lgraph,"addition_5_7","conv_1_6_7");
lgraph = connectLayers(lgraph,"addition_5_7","addition_6_7/in2");
lgraph = connectLayers(lgraph,"conv_2_6_7","maxpool_6_7");
lgraph = connectLayers(lgraph,"conv_2_6_7","mul_1_6_7/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_6_7","mul_1_6_7/in1");
lgraph = connectLayers(lgraph,"mul_1_6_7","addition_6_7/in1");
lgraph = connectLayers(lgraph,"addition_6_7","conv_1_7_7");
lgraph = connectLayers(lgraph,"addition_6_7","addition_7_7/in2");
lgraph = connectLayers(lgraph,"conv_2_7_7","maxpool_7_7");
lgraph = connectLayers(lgraph,"conv_2_7_7","mul_1_7_7/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_7_7","mul_1_7_7/in1");
lgraph = connectLayers(lgraph,"mul_1_7_7","addition_7_7/in1");
lgraph = connectLayers(lgraph,"addition_7_7","conv_1_8_7");
lgraph = connectLayers(lgraph,"addition_7_7","addition_8_7/in2");
lgraph = connectLayers(lgraph,"conv_2_8_7","maxpool_8_7");
lgraph = connectLayers(lgraph,"conv_2_8_7","mul_1_8_7/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_8_7","mul_1_8_7/in1");
lgraph = connectLayers(lgraph,"mul_1_8_7","addition_8_7/in1");
lgraph = connectLayers(lgraph,"addition_8_7","conv_1_9_7");
lgraph = connectLayers(lgraph,"addition_8_7","addition_9_7/in2");
lgraph = connectLayers(lgraph,"conv_2_9_7","maxpool_9_7");
lgraph = connectLayers(lgraph,"conv_2_9_7","mul_1_9_7/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_9_7","mul_1_9_7/in1");
lgraph = connectLayers(lgraph,"mul_1_9_7","addition_9_7/in1");
lgraph = connectLayers(lgraph,"addition_9_7","conv_1_10_7");
lgraph = connectLayers(lgraph,"addition_9_7","addition_10_7/in2");
lgraph = connectLayers(lgraph,"conv_2_10_7","maxpool_10_7");
lgraph = connectLayers(lgraph,"conv_2_10_7","mul_1_10_7/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_10_7","mul_1_10_7/in1");
lgraph = connectLayers(lgraph,"mul_1_10_7","addition_10_7/in1");
lgraph = connectLayers(lgraph,"addition_10_7","conv_1_11_7");
lgraph = connectLayers(lgraph,"addition_10_7","addition_11_7/in2");
lgraph = connectLayers(lgraph,"conv_2_11_7","maxpool_11_7");
lgraph = connectLayers(lgraph,"conv_2_11_7","mul_1_11_7/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_11_7","mul_1_11_7/in1");
lgraph = connectLayers(lgraph,"mul_1_11_7","addition_11_7/in1");
lgraph = connectLayers(lgraph,"addition_11_7","conv_1_12_7");
lgraph = connectLayers(lgraph,"addition_11_7","addition_12_7/in2");
lgraph = connectLayers(lgraph,"conv_2_12_7","maxpool_12_7");
lgraph = connectLayers(lgraph,"conv_2_12_7","mul_1_12_7/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_12_7","mul_1_12_7/in1");
lgraph = connectLayers(lgraph,"mul_1_12_7","addition_12_7/in1");
lgraph = connectLayers(lgraph,"addition_12_7","conv_1_13_7");
lgraph = connectLayers(lgraph,"addition_12_7","addition_13_7/in2");
lgraph = connectLayers(lgraph,"conv_2_13_7","maxpool_13_7");
lgraph = connectLayers(lgraph,"conv_2_13_7","mul_1_13_7/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_13_7","mul_1_13_7/in1");
lgraph = connectLayers(lgraph,"mul_1_13_7","addition_13_7/in1");
lgraph = connectLayers(lgraph,"addition_13_7","conv_1_14_7");
lgraph = connectLayers(lgraph,"addition_13_7","addition_14_7/in2");
lgraph = connectLayers(lgraph,"conv_2_14_7","maxpool_14_7");
lgraph = connectLayers(lgraph,"conv_2_14_7","mul_1_14_7/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_14_7","mul_1_14_7/in1");
lgraph = connectLayers(lgraph,"mul_1_14_7","addition_14_7/in1");
lgraph = connectLayers(lgraph,"addition_14_7","conv_1_15_7");
lgraph = connectLayers(lgraph,"addition_14_7","addition_15_7/in2");
lgraph = connectLayers(lgraph,"conv_2_15_7","maxpool_15_7");
lgraph = connectLayers(lgraph,"conv_2_15_7","mul_1_15_7/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_15_7","mul_1_15_7/in1");
lgraph = connectLayers(lgraph,"mul_1_15_7","addition_15_7/in1");
lgraph = connectLayers(lgraph,"addition_15_7","conv_1_16_7");
lgraph = connectLayers(lgraph,"addition_15_7","addition_16_7/in2");
lgraph = connectLayers(lgraph,"conv_2_16_7","maxpool_16_7");
lgraph = connectLayers(lgraph,"conv_2_16_7","mul_1_16_7/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_16_7","mul_1_16_7/in1");
lgraph = connectLayers(lgraph,"mul_1_16_7","addition_16_7/in1");
lgraph = connectLayers(lgraph,"addition_16_7","conv_1_17_7");
lgraph = connectLayers(lgraph,"addition_16_7","addition_17_7/in2");
lgraph = connectLayers(lgraph,"conv_2_17_7","maxpool_17_7");
lgraph = connectLayers(lgraph,"conv_2_17_7","mul_1_17_7/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_17_7","mul_1_17_7/in1");
lgraph = connectLayers(lgraph,"mul_1_17_7","addition_17_7/in1");
lgraph = connectLayers(lgraph,"addition_17_7","conv_1_18_7");
lgraph = connectLayers(lgraph,"addition_17_7","addition_18_7/in2");
lgraph = connectLayers(lgraph,"conv_2_18_7","maxpool_18_7");
lgraph = connectLayers(lgraph,"conv_2_18_7","mul_1_18_7/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_18_7","mul_1_18_7/in1");
lgraph = connectLayers(lgraph,"mul_1_18_7","addition_18_7/in1");
lgraph = connectLayers(lgraph,"addition_18_7","conv_1_19_7");
lgraph = connectLayers(lgraph,"addition_18_7","addition_19_7/in2");
lgraph = connectLayers(lgraph,"conv_2_19_7","maxpool_19_7");
lgraph = connectLayers(lgraph,"conv_2_19_7","mul_1_19_7/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_19_7","mul_1_19_7/in1");
lgraph = connectLayers(lgraph,"mul_1_19_7","addition_19_7/in1");
lgraph = connectLayers(lgraph,"addition_19_7","conv_1_20_7");
lgraph = connectLayers(lgraph,"addition_19_7","addition_20_7/in2");
lgraph = connectLayers(lgraph,"conv_2_20_7","maxpool_20_7");
lgraph = connectLayers(lgraph,"conv_2_20_7","mul_1_20_7/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_20_7","mul_1_20_7/in1");
lgraph = connectLayers(lgraph,"mul_1_20_7","addition_20_7/in1");

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_1_8")
    reluLayer("Name","relu_1_1_8")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_1_8")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_1_8")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_1_8")
    reluLayer("Name","relu_2_1_8")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_1_8")
    helperSigmoidLayer("sigmoid_1_1_8","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_1_8",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_1_8");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_2_8")
    reluLayer("Name","relu_1_2_8")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_2_8")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_2_8")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_2_8")
    reluLayer("Name","relu_2_2_8")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_2_8")
    helperSigmoidLayer("sigmoid_1_2_8","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_2_8",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_2_8");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_3_8")
    reluLayer("Name","relu_1_3_8")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_3_8")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_3_8")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_3_8")
    reluLayer("Name","relu_2_3_8")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_3_8")
    helperSigmoidLayer("sigmoid_1_3_8","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_3_8",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_3_8");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_4_8")
    reluLayer("Name","relu_1_4_8")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_4_8")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_4_8")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_4_8")
    reluLayer("Name","relu_2_4_8")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_4_8")
    helperSigmoidLayer("sigmoid_1_4_8","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_4_8",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_4_8");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_5_8")
    reluLayer("Name","relu_1_5_8")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_5_8")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_5_8")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_5_8")
    reluLayer("Name","relu_2_5_8")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_5_8")
    helperSigmoidLayer("sigmoid_1_5_8","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_5_8",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_5_8");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_6_8")
    reluLayer("Name","relu_1_6_8")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_6_8")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_6_8")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_6_8")
    reluLayer("Name","relu_2_6_8")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_6_8")
    helperSigmoidLayer("sigmoid_1_6_8","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_6_8",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_6_8");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_7_8")
    reluLayer("Name","relu_1_7_8")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_7_8")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_7_8")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_7_8")
    reluLayer("Name","relu_2_7_8")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_7_8")
    helperSigmoidLayer("sigmoid_1_7_8","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_7_8",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_7_8");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_8_8")
    reluLayer("Name","relu_1_8_8")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_8_8")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_8_8")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_8_8")
    reluLayer("Name","relu_2_8_8")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_8_8")
    helperSigmoidLayer("sigmoid_1_8_8","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_8_8",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_8_8");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_9_8")
    reluLayer("Name","relu_1_9_8")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_9_8")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_9_8")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_9_8")
    reluLayer("Name","relu_2_9_8")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_9_8")
    helperSigmoidLayer("sigmoid_1_9_8","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_9_8",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_9_8");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_10_8")
    reluLayer("Name","relu_1_10_8")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_10_8")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_10_8")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_10_8")
    reluLayer("Name","relu_2_10_8")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_10_8")
    helperSigmoidLayer("sigmoid_1_10_8","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_10_8",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_10_8");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_11_8")
    reluLayer("Name","relu_1_11_8")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_11_8")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_11_8")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_11_8")
    reluLayer("Name","relu_2_11_8")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_11_8")
    helperSigmoidLayer("sigmoid_1_11_8","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_11_8",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_11_8");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_12_8")
    reluLayer("Name","relu_1_12_8")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_12_8")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_12_8")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_12_8")
    reluLayer("Name","relu_2_12_8")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_12_8")
    helperSigmoidLayer("sigmoid_1_12_8","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_12_8",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_12_8");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_13_8")
    reluLayer("Name","relu_1_13_8")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_13_8")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_13_8")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_13_8")
    reluLayer("Name","relu_2_13_8")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_13_8")
    helperSigmoidLayer("sigmoid_1_13_8","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_13_8",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_13_8");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_14_8")
    reluLayer("Name","relu_1_14_8")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_14_8")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_14_8")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_14_8")
    reluLayer("Name","relu_2_14_8")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_14_8")
    helperSigmoidLayer("sigmoid_1_14_8","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_14_8",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_14_8");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_15_8")
    reluLayer("Name","relu_1_15_8")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_15_8")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_15_8")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_15_8")
    reluLayer("Name","relu_2_15_8")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_15_8")
    helperSigmoidLayer("sigmoid_1_15_8","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_15_8",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_15_8");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_16_8")
    reluLayer("Name","relu_1_16_8")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_16_8")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_16_8")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_16_8")
    reluLayer("Name","relu_2_16_8")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_16_8")
    helperSigmoidLayer("sigmoid_1_16_8","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_16_8",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_16_8");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_17_8")
    reluLayer("Name","relu_1_17_8")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_17_8")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_17_8")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_17_8")
    reluLayer("Name","relu_2_17_8")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_17_8")
    helperSigmoidLayer("sigmoid_1_17_8","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_17_8",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_17_8");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_18_8")
    reluLayer("Name","relu_1_18_8")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_18_8")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_18_8")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_18_8")
    reluLayer("Name","relu_2_18_8")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_18_8")
    helperSigmoidLayer("sigmoid_1_18_8","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_18_8",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_18_8");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_19_8")
    reluLayer("Name","relu_1_19_8")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_19_8")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_19_8")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_19_8")
    reluLayer("Name","relu_2_19_8")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_19_8")
    helperSigmoidLayer("sigmoid_1_19_8","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_19_8",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_19_8");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_20_8")
    reluLayer("Name","relu_1_20_8")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_20_8")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_20_8")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_20_8")
    reluLayer("Name","relu_2_20_8")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_20_8")
    helperSigmoidLayer("sigmoid_1_20_8","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_20_8",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","addition_20_8")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_8")];
lgraph = addLayers(lgraph,tempLayers);
l = [additionLayer(2,'Name','sum_1_8')];
lgraph = addLayers(lgraph,l);
lgraph = connectLayers(lgraph,'sum_1_7','sum_1_8/in1');
lgraph = connectLayers(lgraph,'conv_8','sum_1_8/in2');
lgraph = connectLayers(lgraph,"sum_1_7","conv_1_1_8");
lgraph = connectLayers(lgraph,"sum_1_7","addition_1_8/in2");
lgraph = connectLayers(lgraph,"conv_2_1_8","maxpool_1_8");
lgraph = connectLayers(lgraph,"conv_2_1_8","mul_1_1_8/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_1_8","mul_1_1_8/in1");
lgraph = connectLayers(lgraph,"mul_1_1_8","addition_1_8/in1");
lgraph = connectLayers(lgraph,"addition_1_8","conv_1_2_8");
lgraph = connectLayers(lgraph,"addition_1_8","addition_2_8/in2");
lgraph = connectLayers(lgraph,"conv_2_2_8","maxpool_2_8");
lgraph = connectLayers(lgraph,"conv_2_2_8","mul_1_2_8/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_2_8","mul_1_2_8/in1");
lgraph = connectLayers(lgraph,"mul_1_2_8","addition_2_8/in1");
lgraph = connectLayers(lgraph,"addition_2_8","conv_1_3_8");
lgraph = connectLayers(lgraph,"addition_2_8","addition_3_8/in2");
lgraph = connectLayers(lgraph,"conv_2_3_8","maxpool_3_8");
lgraph = connectLayers(lgraph,"conv_2_3_8","mul_1_3_8/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_3_8","mul_1_3_8/in1");
lgraph = connectLayers(lgraph,"mul_1_3_8","addition_3_8/in1");
lgraph = connectLayers(lgraph,"addition_3_8","conv_1_4_8");
lgraph = connectLayers(lgraph,"addition_3_8","addition_4_8/in2");
lgraph = connectLayers(lgraph,"conv_2_4_8","maxpool_4_8");
lgraph = connectLayers(lgraph,"conv_2_4_8","mul_1_4_8/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_4_8","mul_1_4_8/in1");
lgraph = connectLayers(lgraph,"mul_1_4_8","addition_4_8/in1");
lgraph = connectLayers(lgraph,"addition_4_8","conv_1_5_8");
lgraph = connectLayers(lgraph,"addition_4_8","addition_5_8/in2");
lgraph = connectLayers(lgraph,"conv_2_5_8","maxpool_5_8");
lgraph = connectLayers(lgraph,"conv_2_5_8","mul_1_5_8/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_5_8","mul_1_5_8/in1");
lgraph = connectLayers(lgraph,"mul_1_5_8","addition_5_8/in1");
lgraph = connectLayers(lgraph,"addition_5_8","conv_1_6_8");
lgraph = connectLayers(lgraph,"addition_5_8","addition_6_8/in2");
lgraph = connectLayers(lgraph,"conv_2_6_8","maxpool_6_8");
lgraph = connectLayers(lgraph,"conv_2_6_8","mul_1_6_8/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_6_8","mul_1_6_8/in1");
lgraph = connectLayers(lgraph,"mul_1_6_8","addition_6_8/in1");
lgraph = connectLayers(lgraph,"addition_6_8","conv_1_7_8");
lgraph = connectLayers(lgraph,"addition_6_8","addition_7_8/in2");
lgraph = connectLayers(lgraph,"conv_2_7_8","maxpool_7_8");
lgraph = connectLayers(lgraph,"conv_2_7_8","mul_1_7_8/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_7_8","mul_1_7_8/in1");
lgraph = connectLayers(lgraph,"mul_1_7_8","addition_7_8/in1");
lgraph = connectLayers(lgraph,"addition_7_8","conv_1_8_8");
lgraph = connectLayers(lgraph,"addition_7_8","addition_8_8/in2");
lgraph = connectLayers(lgraph,"conv_2_8_8","maxpool_8_8");
lgraph = connectLayers(lgraph,"conv_2_8_8","mul_1_8_8/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_8_8","mul_1_8_8/in1");
lgraph = connectLayers(lgraph,"mul_1_8_8","addition_8_8/in1");
lgraph = connectLayers(lgraph,"addition_8_8","conv_1_9_8");
lgraph = connectLayers(lgraph,"addition_8_8","addition_9_8/in2");
lgraph = connectLayers(lgraph,"conv_2_9_8","maxpool_9_8");
lgraph = connectLayers(lgraph,"conv_2_9_8","mul_1_9_8/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_9_8","mul_1_9_8/in1");
lgraph = connectLayers(lgraph,"mul_1_9_8","addition_9_8/in1");
lgraph = connectLayers(lgraph,"addition_9_8","conv_1_10_8");
lgraph = connectLayers(lgraph,"addition_9_8","addition_10_8/in2");
lgraph = connectLayers(lgraph,"conv_2_10_8","maxpool_10_8");
lgraph = connectLayers(lgraph,"conv_2_10_8","mul_1_10_8/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_10_8","mul_1_10_8/in1");
lgraph = connectLayers(lgraph,"mul_1_10_8","addition_10_8/in1");
lgraph = connectLayers(lgraph,"addition_10_8","conv_1_11_8");
lgraph = connectLayers(lgraph,"addition_10_8","addition_11_8/in2");
lgraph = connectLayers(lgraph,"conv_2_11_8","maxpool_11_8");
lgraph = connectLayers(lgraph,"conv_2_11_8","mul_1_11_8/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_11_8","mul_1_11_8/in1");
lgraph = connectLayers(lgraph,"mul_1_11_8","addition_11_8/in1");
lgraph = connectLayers(lgraph,"addition_11_8","conv_1_12_8");
lgraph = connectLayers(lgraph,"addition_11_8","addition_12_8/in2");
lgraph = connectLayers(lgraph,"conv_2_12_8","maxpool_12_8");
lgraph = connectLayers(lgraph,"conv_2_12_8","mul_1_12_8/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_12_8","mul_1_12_8/in1");
lgraph = connectLayers(lgraph,"mul_1_12_8","addition_12_8/in1");
lgraph = connectLayers(lgraph,"addition_12_8","conv_1_13_8");
lgraph = connectLayers(lgraph,"addition_12_8","addition_13_8/in2");
lgraph = connectLayers(lgraph,"conv_2_13_8","maxpool_13_8");
lgraph = connectLayers(lgraph,"conv_2_13_8","mul_1_13_8/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_13_8","mul_1_13_8/in1");
lgraph = connectLayers(lgraph,"mul_1_13_8","addition_13_8/in1");
lgraph = connectLayers(lgraph,"addition_13_8","conv_1_14_8");
lgraph = connectLayers(lgraph,"addition_13_8","addition_14_8/in2");
lgraph = connectLayers(lgraph,"conv_2_14_8","maxpool_14_8");
lgraph = connectLayers(lgraph,"conv_2_14_8","mul_1_14_8/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_14_8","mul_1_14_8/in1");
lgraph = connectLayers(lgraph,"mul_1_14_8","addition_14_8/in1");
lgraph = connectLayers(lgraph,"addition_14_8","conv_1_15_8");
lgraph = connectLayers(lgraph,"addition_14_8","addition_15_8/in2");
lgraph = connectLayers(lgraph,"conv_2_15_8","maxpool_15_8");
lgraph = connectLayers(lgraph,"conv_2_15_8","mul_1_15_8/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_15_8","mul_1_15_8/in1");
lgraph = connectLayers(lgraph,"mul_1_15_8","addition_15_8/in1");
lgraph = connectLayers(lgraph,"addition_15_8","conv_1_16_8");
lgraph = connectLayers(lgraph,"addition_15_8","addition_16_8/in2");
lgraph = connectLayers(lgraph,"conv_2_16_8","maxpool_16_8");
lgraph = connectLayers(lgraph,"conv_2_16_8","mul_1_16_8/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_16_8","mul_1_16_8/in1");
lgraph = connectLayers(lgraph,"mul_1_16_8","addition_16_8/in1");
lgraph = connectLayers(lgraph,"addition_16_8","conv_1_17_8");
lgraph = connectLayers(lgraph,"addition_16_8","addition_17_8/in2");
lgraph = connectLayers(lgraph,"conv_2_17_8","maxpool_17_8");
lgraph = connectLayers(lgraph,"conv_2_17_8","mul_1_17_8/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_17_8","mul_1_17_8/in1");
lgraph = connectLayers(lgraph,"mul_1_17_8","addition_17_8/in1");
lgraph = connectLayers(lgraph,"addition_17_8","conv_1_18_8");
lgraph = connectLayers(lgraph,"addition_17_8","addition_18_8/in2");
lgraph = connectLayers(lgraph,"conv_2_18_8","maxpool_18_8");
lgraph = connectLayers(lgraph,"conv_2_18_8","mul_1_18_8/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_18_8","mul_1_18_8/in1");
lgraph = connectLayers(lgraph,"mul_1_18_8","addition_18_8/in1");
lgraph = connectLayers(lgraph,"addition_18_8","conv_1_19_8");
lgraph = connectLayers(lgraph,"addition_18_8","addition_19_8/in2");
lgraph = connectLayers(lgraph,"conv_2_19_8","maxpool_19_8");
lgraph = connectLayers(lgraph,"conv_2_19_8","mul_1_19_8/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_19_8","mul_1_19_8/in1");
lgraph = connectLayers(lgraph,"mul_1_19_8","addition_19_8/in1");
lgraph = connectLayers(lgraph,"addition_19_8","conv_1_20_8");
lgraph = connectLayers(lgraph,"addition_19_8","addition_20_8/in2");
lgraph = connectLayers(lgraph,"conv_2_20_8","maxpool_20_8");
lgraph = connectLayers(lgraph,"conv_2_20_8","mul_1_20_8/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_20_8","mul_1_20_8/in1");
lgraph = connectLayers(lgraph,"mul_1_20_8","addition_20_8/in1");

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_1_9")
    reluLayer("Name","relu_1_1_9")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_1_9")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_1_9")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_1_9")
    reluLayer("Name","relu_2_1_9")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_1_9")
    helperSigmoidLayer("sigmoid_1_1_9","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_1_9",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_1_9");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_2_9")
    reluLayer("Name","relu_1_2_9")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_2_9")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_2_9")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_2_9")
    reluLayer("Name","relu_2_2_9")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_2_9")
    helperSigmoidLayer("sigmoid_1_2_9","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_2_9",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_2_9");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_3_9")
    reluLayer("Name","relu_1_3_9")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_3_9")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_3_9")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_3_9")
    reluLayer("Name","relu_2_3_9")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_3_9")
    helperSigmoidLayer("sigmoid_1_3_9","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_3_9",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_3_9");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_4_9")
    reluLayer("Name","relu_1_4_9")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_4_9")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_4_9")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_4_9")
    reluLayer("Name","relu_2_4_9")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_4_9")
    helperSigmoidLayer("sigmoid_1_4_9","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_4_9",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_4_9");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_5_9")
    reluLayer("Name","relu_1_5_9")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_5_9")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_5_9")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_5_9")
    reluLayer("Name","relu_2_5_9")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_5_9")
    helperSigmoidLayer("sigmoid_1_5_9","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_5_9",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_5_9");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_6_9")
    reluLayer("Name","relu_1_6_9")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_6_9")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_6_9")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_6_9")
    reluLayer("Name","relu_2_6_9")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_6_9")
    helperSigmoidLayer("sigmoid_1_6_9","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_6_9",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_6_9");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_7_9")
    reluLayer("Name","relu_1_7_9")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_7_9")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_7_9")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_7_9")
    reluLayer("Name","relu_2_7_9")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_7_9")
    helperSigmoidLayer("sigmoid_1_7_9","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_7_9",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_7_9");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_8_9")
    reluLayer("Name","relu_1_8_9")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_8_9")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_8_9")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_8_9")
    reluLayer("Name","relu_2_8_9")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_8_9")
    helperSigmoidLayer("sigmoid_1_8_9","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_8_9",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_8_9");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_9_9")
    reluLayer("Name","relu_1_9_9")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_9_9")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_9_9")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_9_9")
    reluLayer("Name","relu_2_9_9")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_9_9")
    helperSigmoidLayer("sigmoid_1_9_9","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_9_9",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_9_9");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_10_9")
    reluLayer("Name","relu_1_10_9")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_10_9")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_10_9")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_10_9")
    reluLayer("Name","relu_2_10_9")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_10_9")
    helperSigmoidLayer("sigmoid_1_10_9","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_10_9",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_10_9");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_11_9")
    reluLayer("Name","relu_1_11_9")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_11_9")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_11_9")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_11_9")
    reluLayer("Name","relu_2_11_9")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_11_9")
    helperSigmoidLayer("sigmoid_1_11_9","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_11_9",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_11_9");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_12_9")
    reluLayer("Name","relu_1_12_9")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_12_9")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_12_9")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_12_9")
    reluLayer("Name","relu_2_12_9")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_12_9")
    helperSigmoidLayer("sigmoid_1_12_9","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_12_9",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_12_9");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_13_9")
    reluLayer("Name","relu_1_13_9")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_13_9")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_13_9")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_13_9")
    reluLayer("Name","relu_2_13_9")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_13_9")
    helperSigmoidLayer("sigmoid_1_13_9","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_13_9",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_13_9");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_14_9")
    reluLayer("Name","relu_1_14_9")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_14_9")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_14_9")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_14_9")
    reluLayer("Name","relu_2_14_9")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_14_9")
    helperSigmoidLayer("sigmoid_1_14_9","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_14_9",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_14_9");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_15_9")
    reluLayer("Name","relu_1_15_9")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_15_9")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_15_9")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_15_9")
    reluLayer("Name","relu_2_15_9")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_15_9")
    helperSigmoidLayer("sigmoid_1_15_9","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_15_9",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_15_9");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_16_9")
    reluLayer("Name","relu_1_16_9")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_16_9")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_16_9")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_16_9")
    reluLayer("Name","relu_2_16_9")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_16_9")
    helperSigmoidLayer("sigmoid_1_16_9","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_16_9",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_16_9");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_17_9")
    reluLayer("Name","relu_1_17_9")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_17_9")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_17_9")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_17_9")
    reluLayer("Name","relu_2_17_9")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_17_9")
    helperSigmoidLayer("sigmoid_1_17_9","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_17_9",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_17_9");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_18_9")
    reluLayer("Name","relu_1_18_9")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_18_9")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_18_9")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_18_9")
    reluLayer("Name","relu_2_18_9")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_18_9")
    helperSigmoidLayer("sigmoid_1_18_9","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_18_9",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_18_9");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_19_9")
    reluLayer("Name","relu_1_19_9")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_19_9")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_19_9")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_19_9")
    reluLayer("Name","relu_2_19_9")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_19_9")
    helperSigmoidLayer("sigmoid_1_19_9","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_19_9",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_19_9");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_20_9")
    reluLayer("Name","relu_1_20_9")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_20_9")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_20_9")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_20_9")
    reluLayer("Name","relu_2_20_9")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_20_9")
    helperSigmoidLayer("sigmoid_1_20_9","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_20_9",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","addition_20_9")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_9")];
lgraph = addLayers(lgraph,tempLayers);
l = [additionLayer(2,'Name','sum_1_9')];
lgraph = addLayers(lgraph,l);
lgraph = connectLayers(lgraph,'sum_1_8','sum_1_9/in1');
lgraph = connectLayers(lgraph,'conv_9','sum_1_9/in2');
lgraph = connectLayers(lgraph,"sum_1_8","conv_1_1_9");
lgraph = connectLayers(lgraph,"sum_1_8","addition_1_9/in2");
lgraph = connectLayers(lgraph,"conv_2_1_9","maxpool_1_9");
lgraph = connectLayers(lgraph,"conv_2_1_9","mul_1_1_9/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_1_9","mul_1_1_9/in1");
lgraph = connectLayers(lgraph,"mul_1_1_9","addition_1_9/in1");
lgraph = connectLayers(lgraph,"addition_1_9","conv_1_2_9");
lgraph = connectLayers(lgraph,"addition_1_9","addition_2_9/in2");
lgraph = connectLayers(lgraph,"conv_2_2_9","maxpool_2_9");
lgraph = connectLayers(lgraph,"conv_2_2_9","mul_1_2_9/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_2_9","mul_1_2_9/in1");
lgraph = connectLayers(lgraph,"mul_1_2_9","addition_2_9/in1");
lgraph = connectLayers(lgraph,"addition_2_9","conv_1_3_9");
lgraph = connectLayers(lgraph,"addition_2_9","addition_3_9/in2");
lgraph = connectLayers(lgraph,"conv_2_3_9","maxpool_3_9");
lgraph = connectLayers(lgraph,"conv_2_3_9","mul_1_3_9/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_3_9","mul_1_3_9/in1");
lgraph = connectLayers(lgraph,"mul_1_3_9","addition_3_9/in1");
lgraph = connectLayers(lgraph,"addition_3_9","conv_1_4_9");
lgraph = connectLayers(lgraph,"addition_3_9","addition_4_9/in2");
lgraph = connectLayers(lgraph,"conv_2_4_9","maxpool_4_9");
lgraph = connectLayers(lgraph,"conv_2_4_9","mul_1_4_9/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_4_9","mul_1_4_9/in1");
lgraph = connectLayers(lgraph,"mul_1_4_9","addition_4_9/in1");
lgraph = connectLayers(lgraph,"addition_4_9","conv_1_5_9");
lgraph = connectLayers(lgraph,"addition_4_9","addition_5_9/in2");
lgraph = connectLayers(lgraph,"conv_2_5_9","maxpool_5_9");
lgraph = connectLayers(lgraph,"conv_2_5_9","mul_1_5_9/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_5_9","mul_1_5_9/in1");
lgraph = connectLayers(lgraph,"mul_1_5_9","addition_5_9/in1");
lgraph = connectLayers(lgraph,"addition_5_9","conv_1_6_9");
lgraph = connectLayers(lgraph,"addition_5_9","addition_6_9/in2");
lgraph = connectLayers(lgraph,"conv_2_6_9","maxpool_6_9");
lgraph = connectLayers(lgraph,"conv_2_6_9","mul_1_6_9/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_6_9","mul_1_6_9/in1");
lgraph = connectLayers(lgraph,"mul_1_6_9","addition_6_9/in1");
lgraph = connectLayers(lgraph,"addition_6_9","conv_1_7_9");
lgraph = connectLayers(lgraph,"addition_6_9","addition_7_9/in2");
lgraph = connectLayers(lgraph,"conv_2_7_9","maxpool_7_9");
lgraph = connectLayers(lgraph,"conv_2_7_9","mul_1_7_9/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_7_9","mul_1_7_9/in1");
lgraph = connectLayers(lgraph,"mul_1_7_9","addition_7_9/in1");
lgraph = connectLayers(lgraph,"addition_7_9","conv_1_8_9");
lgraph = connectLayers(lgraph,"addition_7_9","addition_8_9/in2");
lgraph = connectLayers(lgraph,"conv_2_8_9","maxpool_8_9");
lgraph = connectLayers(lgraph,"conv_2_8_9","mul_1_8_9/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_8_9","mul_1_8_9/in1");
lgraph = connectLayers(lgraph,"mul_1_8_9","addition_8_9/in1");
lgraph = connectLayers(lgraph,"addition_8_9","conv_1_9_9");
lgraph = connectLayers(lgraph,"addition_8_9","addition_9_9/in2");
lgraph = connectLayers(lgraph,"conv_2_9_9","maxpool_9_9");
lgraph = connectLayers(lgraph,"conv_2_9_9","mul_1_9_9/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_9_9","mul_1_9_9/in1");
lgraph = connectLayers(lgraph,"mul_1_9_9","addition_9_9/in1");
lgraph = connectLayers(lgraph,"addition_9_9","conv_1_10_9");
lgraph = connectLayers(lgraph,"addition_9_9","addition_10_9/in2");
lgraph = connectLayers(lgraph,"conv_2_10_9","maxpool_10_9");
lgraph = connectLayers(lgraph,"conv_2_10_9","mul_1_10_9/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_10_9","mul_1_10_9/in1");
lgraph = connectLayers(lgraph,"mul_1_10_9","addition_10_9/in1");
lgraph = connectLayers(lgraph,"addition_10_9","conv_1_11_9");
lgraph = connectLayers(lgraph,"addition_10_9","addition_11_9/in2");
lgraph = connectLayers(lgraph,"conv_2_11_9","maxpool_11_9");
lgraph = connectLayers(lgraph,"conv_2_11_9","mul_1_11_9/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_11_9","mul_1_11_9/in1");
lgraph = connectLayers(lgraph,"mul_1_11_9","addition_11_9/in1");
lgraph = connectLayers(lgraph,"addition_11_9","conv_1_12_9");
lgraph = connectLayers(lgraph,"addition_11_9","addition_12_9/in2");
lgraph = connectLayers(lgraph,"conv_2_12_9","maxpool_12_9");
lgraph = connectLayers(lgraph,"conv_2_12_9","mul_1_12_9/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_12_9","mul_1_12_9/in1");
lgraph = connectLayers(lgraph,"mul_1_12_9","addition_12_9/in1");
lgraph = connectLayers(lgraph,"addition_12_9","conv_1_13_9");
lgraph = connectLayers(lgraph,"addition_12_9","addition_13_9/in2");
lgraph = connectLayers(lgraph,"conv_2_13_9","maxpool_13_9");
lgraph = connectLayers(lgraph,"conv_2_13_9","mul_1_13_9/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_13_9","mul_1_13_9/in1");
lgraph = connectLayers(lgraph,"mul_1_13_9","addition_13_9/in1");
lgraph = connectLayers(lgraph,"addition_13_9","conv_1_14_9");
lgraph = connectLayers(lgraph,"addition_13_9","addition_14_9/in2");
lgraph = connectLayers(lgraph,"conv_2_14_9","maxpool_14_9");
lgraph = connectLayers(lgraph,"conv_2_14_9","mul_1_14_9/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_14_9","mul_1_14_9/in1");
lgraph = connectLayers(lgraph,"mul_1_14_9","addition_14_9/in1");
lgraph = connectLayers(lgraph,"addition_14_9","conv_1_15_9");
lgraph = connectLayers(lgraph,"addition_14_9","addition_15_9/in2");
lgraph = connectLayers(lgraph,"conv_2_15_9","maxpool_15_9");
lgraph = connectLayers(lgraph,"conv_2_15_9","mul_1_15_9/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_15_9","mul_1_15_9/in1");
lgraph = connectLayers(lgraph,"mul_1_15_9","addition_15_9/in1");
lgraph = connectLayers(lgraph,"addition_15_9","conv_1_16_9");
lgraph = connectLayers(lgraph,"addition_15_9","addition_16_9/in2");
lgraph = connectLayers(lgraph,"conv_2_16_9","maxpool_16_9");
lgraph = connectLayers(lgraph,"conv_2_16_9","mul_1_16_9/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_16_9","mul_1_16_9/in1");
lgraph = connectLayers(lgraph,"mul_1_16_9","addition_16_9/in1");
lgraph = connectLayers(lgraph,"addition_16_9","conv_1_17_9");
lgraph = connectLayers(lgraph,"addition_16_9","addition_17_9/in2");
lgraph = connectLayers(lgraph,"conv_2_17_9","maxpool_17_9");
lgraph = connectLayers(lgraph,"conv_2_17_9","mul_1_17_9/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_17_9","mul_1_17_9/in1");
lgraph = connectLayers(lgraph,"mul_1_17_9","addition_17_9/in1");
lgraph = connectLayers(lgraph,"addition_17_9","conv_1_18_9");
lgraph = connectLayers(lgraph,"addition_17_9","addition_18_9/in2");
lgraph = connectLayers(lgraph,"conv_2_18_9","maxpool_18_9");
lgraph = connectLayers(lgraph,"conv_2_18_9","mul_1_18_9/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_18_9","mul_1_18_9/in1");
lgraph = connectLayers(lgraph,"mul_1_18_9","addition_18_9/in1");
lgraph = connectLayers(lgraph,"addition_18_9","conv_1_19_9");
lgraph = connectLayers(lgraph,"addition_18_9","addition_19_9/in2");
lgraph = connectLayers(lgraph,"conv_2_19_9","maxpool_19_9");
lgraph = connectLayers(lgraph,"conv_2_19_9","mul_1_19_9/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_19_9","mul_1_19_9/in1");
lgraph = connectLayers(lgraph,"mul_1_19_9","addition_19_9/in1");
lgraph = connectLayers(lgraph,"addition_19_9","conv_1_20_9");
lgraph = connectLayers(lgraph,"addition_19_9","addition_20_9/in2");
lgraph = connectLayers(lgraph,"conv_2_20_9","maxpool_20_9");
lgraph = connectLayers(lgraph,"conv_2_20_9","mul_1_20_9/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_20_9","mul_1_20_9/in1");
lgraph = connectLayers(lgraph,"mul_1_20_9","addition_20_9/in1");

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_1_10")
    reluLayer("Name","relu_1_1_10")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_1_10")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_1_10")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_1_10")
    reluLayer("Name","relu_2_1_10")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_1_10")
    helperSigmoidLayer("sigmoid_1_1_10","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_1_10",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_1_10");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_2_10")
    reluLayer("Name","relu_1_2_10")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_2_10")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_2_10")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_2_10")
    reluLayer("Name","relu_2_2_10")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_2_10")
    helperSigmoidLayer("sigmoid_1_2_10","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_2_10",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_2_10");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_3_10")
    reluLayer("Name","relu_1_3_10")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_3_10")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_3_10")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_3_10")
    reluLayer("Name","relu_2_3_10")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_3_10")
    helperSigmoidLayer("sigmoid_1_3_10","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_3_10",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_3_10");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_4_10")
    reluLayer("Name","relu_1_4_10")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_4_10")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_4_10")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_4_10")
    reluLayer("Name","relu_2_4_10")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_4_10")
    helperSigmoidLayer("sigmoid_1_4_10","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_4_10",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_4_10");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_5_10")
    reluLayer("Name","relu_1_5_10")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_5_10")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_5_10")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_5_10")
    reluLayer("Name","relu_2_5_10")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_5_10")
    helperSigmoidLayer("sigmoid_1_5_10","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_5_10",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_5_10");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_6_10")
    reluLayer("Name","relu_1_6_10")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_6_10")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_6_10")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_6_10")
    reluLayer("Name","relu_2_6_10")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_6_10")
    helperSigmoidLayer("sigmoid_1_6_10","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_6_10",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_6_10");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_7_10")
    reluLayer("Name","relu_1_7_10")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_7_10")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_7_10")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_7_10")
    reluLayer("Name","relu_2_7_10")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_7_10")
    helperSigmoidLayer("sigmoid_1_7_10","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_7_10",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_7_10");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_8_10")
    reluLayer("Name","relu_1_8_10")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_8_10")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_8_10")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_8_10")
    reluLayer("Name","relu_2_8_10")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_8_10")
    helperSigmoidLayer("sigmoid_1_8_10","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_8_10",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_8_10");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_9_10")
    reluLayer("Name","relu_1_9_10")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_9_10")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_9_10")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_9_10")
    reluLayer("Name","relu_2_9_10")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_9_10")
    helperSigmoidLayer("sigmoid_1_9_10","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_9_10",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_9_10");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_10_10")
    reluLayer("Name","relu_1_10_10")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_10_10")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_10_10")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_10_10")
    reluLayer("Name","relu_2_10_10")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_10_10")
    helperSigmoidLayer("sigmoid_1_10_10","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_10_10",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_10_10");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_11_10")
    reluLayer("Name","relu_1_11_10")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_11_10")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_11_10")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_11_10")
    reluLayer("Name","relu_2_11_10")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_11_10")
    helperSigmoidLayer("sigmoid_1_11_10","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_11_10",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_11_10");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_12_10")
    reluLayer("Name","relu_1_12_10")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_12_10")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_12_10")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_12_10")
    reluLayer("Name","relu_2_12_10")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_12_10")
    helperSigmoidLayer("sigmoid_1_12_10","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_12_10",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_12_10");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_13_10")
    reluLayer("Name","relu_1_13_10")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_13_10")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_13_10")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_13_10")
    reluLayer("Name","relu_2_13_10")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_13_10")
    helperSigmoidLayer("sigmoid_1_13_10","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_13_10",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_13_10");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_14_10")
    reluLayer("Name","relu_1_14_10")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_14_10")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_14_10")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_14_10")
    reluLayer("Name","relu_2_14_10")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_14_10")
    helperSigmoidLayer("sigmoid_1_14_10","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_14_10",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_14_10");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_15_10")
    reluLayer("Name","relu_1_15_10")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_15_10")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_15_10")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_15_10")
    reluLayer("Name","relu_2_15_10")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_15_10")
    helperSigmoidLayer("sigmoid_1_15_10","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_15_10",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_15_10");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_16_10")
    reluLayer("Name","relu_1_16_10")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_16_10")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_16_10")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_16_10")
    reluLayer("Name","relu_2_16_10")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_16_10")
    helperSigmoidLayer("sigmoid_1_16_10","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_16_10",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_16_10");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_17_10")
    reluLayer("Name","relu_1_17_10")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_17_10")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_17_10")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_17_10")
    reluLayer("Name","relu_2_17_10")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_17_10")
    helperSigmoidLayer("sigmoid_1_17_10","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_17_10",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_17_10");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_18_10")
    reluLayer("Name","relu_1_18_10")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_18_10")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_18_10")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_18_10")
    reluLayer("Name","relu_2_18_10")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_18_10")
    helperSigmoidLayer("sigmoid_1_18_10","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_18_10",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_18_10");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_19_10")
    reluLayer("Name","relu_1_19_10")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_19_10")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_19_10")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_19_10")
    reluLayer("Name","relu_2_19_10")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_19_10")
    helperSigmoidLayer("sigmoid_1_19_10","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_19_10",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_19_10");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_1_20_10")
    reluLayer("Name","relu_1_20_10")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_2_20_10")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Padding","same","Name","maxpool_20_10")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_3_20_10")
    reluLayer("Name","relu_2_20_10")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_4_20_10")
    helperSigmoidLayer("sigmoid_1_20_10","in","out")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = helperElementWiseMultiplication("mul_1_20_10",["in1" "in2"],"out");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","addition_20_10")
    convolution2dLayer([3 3],64,"Padding","same","Name","conv_10")];
lgraph = addLayers(lgraph,tempLayers);
l = [additionLayer(2,'Name','sum_1_10')];
lgraph = addLayers(lgraph,l);
lgraph = connectLayers(lgraph,'sum_1_9','sum_1_10/in1');
lgraph = connectLayers(lgraph,'conv_10','sum_1_10/in2');
lgraph = connectLayers(lgraph,"sum_1_9","conv_1_1_10");
lgraph = connectLayers(lgraph,"sum_1_9","addition_1_10/in2");
lgraph = connectLayers(lgraph,"conv_2_1_10","maxpool_1_10");
lgraph = connectLayers(lgraph,"conv_2_1_10","mul_1_1_10/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_1_10","mul_1_1_10/in1");
lgraph = connectLayers(lgraph,"mul_1_1_10","addition_1_10/in1");
lgraph = connectLayers(lgraph,"addition_1_10","conv_1_2_10");
lgraph = connectLayers(lgraph,"addition_1_10","addition_2_10/in2");
lgraph = connectLayers(lgraph,"conv_2_2_10","maxpool_2_10");
lgraph = connectLayers(lgraph,"conv_2_2_10","mul_1_2_10/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_2_10","mul_1_2_10/in1");
lgraph = connectLayers(lgraph,"mul_1_2_10","addition_2_10/in1");
lgraph = connectLayers(lgraph,"addition_2_10","conv_1_3_10");
lgraph = connectLayers(lgraph,"addition_2_10","addition_3_10/in2");
lgraph = connectLayers(lgraph,"conv_2_3_10","maxpool_3_10");
lgraph = connectLayers(lgraph,"conv_2_3_10","mul_1_3_10/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_3_10","mul_1_3_10/in1");
lgraph = connectLayers(lgraph,"mul_1_3_10","addition_3_10/in1");
lgraph = connectLayers(lgraph,"addition_3_10","conv_1_4_10");
lgraph = connectLayers(lgraph,"addition_3_10","addition_4_10/in2");
lgraph = connectLayers(lgraph,"conv_2_4_10","maxpool_4_10");
lgraph = connectLayers(lgraph,"conv_2_4_10","mul_1_4_10/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_4_10","mul_1_4_10/in1");
lgraph = connectLayers(lgraph,"mul_1_4_10","addition_4_10/in1");
lgraph = connectLayers(lgraph,"addition_4_10","conv_1_5_10");
lgraph = connectLayers(lgraph,"addition_4_10","addition_5_10/in2");
lgraph = connectLayers(lgraph,"conv_2_5_10","maxpool_5_10");
lgraph = connectLayers(lgraph,"conv_2_5_10","mul_1_5_10/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_5_10","mul_1_5_10/in1");
lgraph = connectLayers(lgraph,"mul_1_5_10","addition_5_10/in1");
lgraph = connectLayers(lgraph,"addition_5_10","conv_1_6_10");
lgraph = connectLayers(lgraph,"addition_5_10","addition_6_10/in2");
lgraph = connectLayers(lgraph,"conv_2_6_10","maxpool_6_10");
lgraph = connectLayers(lgraph,"conv_2_6_10","mul_1_6_10/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_6_10","mul_1_6_10/in1");
lgraph = connectLayers(lgraph,"mul_1_6_10","addition_6_10/in1");
lgraph = connectLayers(lgraph,"addition_6_10","conv_1_7_10");
lgraph = connectLayers(lgraph,"addition_6_10","addition_7_10/in2");
lgraph = connectLayers(lgraph,"conv_2_7_10","maxpool_7_10");
lgraph = connectLayers(lgraph,"conv_2_7_10","mul_1_7_10/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_7_10","mul_1_7_10/in1");
lgraph = connectLayers(lgraph,"mul_1_7_10","addition_7_10/in1");
lgraph = connectLayers(lgraph,"addition_7_10","conv_1_8_10");
lgraph = connectLayers(lgraph,"addition_7_10","addition_8_10/in2");
lgraph = connectLayers(lgraph,"conv_2_8_10","maxpool_8_10");
lgraph = connectLayers(lgraph,"conv_2_8_10","mul_1_8_10/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_8_10","mul_1_8_10/in1");
lgraph = connectLayers(lgraph,"mul_1_8_10","addition_8_10/in1");
lgraph = connectLayers(lgraph,"addition_8_10","conv_1_9_10");
lgraph = connectLayers(lgraph,"addition_8_10","addition_9_10/in2");
lgraph = connectLayers(lgraph,"conv_2_9_10","maxpool_9_10");
lgraph = connectLayers(lgraph,"conv_2_9_10","mul_1_9_10/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_9_10","mul_1_9_10/in1");
lgraph = connectLayers(lgraph,"mul_1_9_10","addition_9_10/in1");
lgraph = connectLayers(lgraph,"addition_9_10","conv_1_10_10");
lgraph = connectLayers(lgraph,"addition_9_10","addition_10_10/in2");
lgraph = connectLayers(lgraph,"conv_2_10_10","maxpool_10_10");
lgraph = connectLayers(lgraph,"conv_2_10_10","mul_1_10_10/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_10_10","mul_1_10_10/in1");
lgraph = connectLayers(lgraph,"mul_1_10_10","addition_10_10/in1");
lgraph = connectLayers(lgraph,"addition_10_10","conv_1_11_10");
lgraph = connectLayers(lgraph,"addition_10_10","addition_11_10/in2");
lgraph = connectLayers(lgraph,"conv_2_11_10","maxpool_11_10");
lgraph = connectLayers(lgraph,"conv_2_11_10","mul_1_11_10/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_11_10","mul_1_11_10/in1");
lgraph = connectLayers(lgraph,"mul_1_11_10","addition_11_10/in1");
lgraph = connectLayers(lgraph,"addition_11_10","conv_1_12_10");
lgraph = connectLayers(lgraph,"addition_11_10","addition_12_10/in2");
lgraph = connectLayers(lgraph,"conv_2_12_10","maxpool_12_10");
lgraph = connectLayers(lgraph,"conv_2_12_10","mul_1_12_10/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_12_10","mul_1_12_10/in1");
lgraph = connectLayers(lgraph,"mul_1_12_10","addition_12_10/in1");
lgraph = connectLayers(lgraph,"addition_12_10","conv_1_13_10");
lgraph = connectLayers(lgraph,"addition_12_10","addition_13_10/in2");
lgraph = connectLayers(lgraph,"conv_2_13_10","maxpool_13_10");
lgraph = connectLayers(lgraph,"conv_2_13_10","mul_1_13_10/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_13_10","mul_1_13_10/in1");
lgraph = connectLayers(lgraph,"mul_1_13_10","addition_13_10/in1");
lgraph = connectLayers(lgraph,"addition_13_10","conv_1_14_10");
lgraph = connectLayers(lgraph,"addition_13_10","addition_14_10/in2");
lgraph = connectLayers(lgraph,"conv_2_14_10","maxpool_14_10");
lgraph = connectLayers(lgraph,"conv_2_14_10","mul_1_14_10/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_14_10","mul_1_14_10/in1");
lgraph = connectLayers(lgraph,"mul_1_14_10","addition_14_10/in1");
lgraph = connectLayers(lgraph,"addition_14_10","conv_1_15_10");
lgraph = connectLayers(lgraph,"addition_14_10","addition_15_10/in2");
lgraph = connectLayers(lgraph,"conv_2_15_10","maxpool_15_10");
lgraph = connectLayers(lgraph,"conv_2_15_10","mul_1_15_10/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_15_10","mul_1_15_10/in1");
lgraph = connectLayers(lgraph,"mul_1_15_10","addition_15_10/in1");
lgraph = connectLayers(lgraph,"addition_15_10","conv_1_16_10");
lgraph = connectLayers(lgraph,"addition_15_10","addition_16_10/in2");
lgraph = connectLayers(lgraph,"conv_2_16_10","maxpool_16_10");
lgraph = connectLayers(lgraph,"conv_2_16_10","mul_1_16_10/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_16_10","mul_1_16_10/in1");
lgraph = connectLayers(lgraph,"mul_1_16_10","addition_16_10/in1");
lgraph = connectLayers(lgraph,"addition_16_10","conv_1_17_10");
lgraph = connectLayers(lgraph,"addition_16_10","addition_17_10/in2");
lgraph = connectLayers(lgraph,"conv_2_17_10","maxpool_17_10");
lgraph = connectLayers(lgraph,"conv_2_17_10","mul_1_17_10/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_17_10","mul_1_17_10/in1");
lgraph = connectLayers(lgraph,"mul_1_17_10","addition_17_10/in1");
lgraph = connectLayers(lgraph,"addition_17_10","conv_1_18_10");
lgraph = connectLayers(lgraph,"addition_17_10","addition_18_10/in2");
lgraph = connectLayers(lgraph,"conv_2_18_10","maxpool_18_10");
lgraph = connectLayers(lgraph,"conv_2_18_10","mul_1_18_10/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_18_10","mul_1_18_10/in1");
lgraph = connectLayers(lgraph,"mul_1_18_10","addition_18_10/in1");
lgraph = connectLayers(lgraph,"addition_18_10","conv_1_19_10");
lgraph = connectLayers(lgraph,"addition_18_10","addition_19_10/in2");
lgraph = connectLayers(lgraph,"conv_2_19_10","maxpool_19_10");
lgraph = connectLayers(lgraph,"conv_2_19_10","mul_1_19_10/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_19_10","mul_1_19_10/in1");
lgraph = connectLayers(lgraph,"mul_1_19_10","addition_19_10/in1");
lgraph = connectLayers(lgraph,"addition_19_10","conv_1_20_10");
lgraph = connectLayers(lgraph,"addition_19_10","addition_20_10/in2");
lgraph = connectLayers(lgraph,"conv_2_20_10","maxpool_20_10");
lgraph = connectLayers(lgraph,"conv_2_20_10","mul_1_20_10/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_20_10","mul_1_20_10/in1");
lgraph = connectLayers(lgraph,"mul_1_20_10","addition_20_10/in1");

tempLayers1 = [
    convolution2dLayer([3 3],64,"Padding","same","Name","con")
    additionLayer(2,"Name","add")];
lgraph = addLayers(lgraph,tempLayers1);
lgraph = connectLayers(lgraph,"sum_1_10","con");
lgraph = connectLayers(lgraph,"conv_start","add/in2");

tempLayers1 = [
    transposedConv2dLayer(1,64,'Stride',2,"Name","decon")
    convolution2dLayer([1 1],3,"Name","con_end")
    clippedReluLayer(1.0,"Name","relu_end")
    regressionLayer("Name","end")];
lgraph = addLayers(lgraph,tempLayers1);
lgraph = connectLayers(lgraph,"add","decon");

options = trainingOptions('adam', ...
    'ExecutionEnvironment','gpu', ...
    'MaxEpochs',1, ...
    'MiniBatchSize',10, ...
    'GradientThreshold',1, ...
    'InitialLearnRate',0.00001, ...
    'SquaredGradientDecayFactor',0.999, ...
    'Epsilon',0.000000001, ...
    'Verbose',false, ...
    'Plots','training-progress');
disp("ok");
net = trainNetwork(imdsCombined,lgraph,options);
modelDateTime = datestr(now,'dd-mmm-yyyy-HH-MM-SS');
save(['trainedNet-' modelDateTime '-Epoch-' num2str(1000) ...
            'ScaleFactors-' num2str(234) '.mat'],'net','options');
clear tempLayers;
plot(lgraph);
end

function layer = helperElementWiseMultiplication(name,~,~)
% Define this function before running the script.
    layer = ElementWiseMultiplication(2,name);
% The function must create and return a layer of type ElementWiseMultiplication.
end
function layer = helperSigmoidLayer(name,~,~)
% Define this function before running the script.
    layer = sigmoidLayer(name);
% The function must create and return a layer of type sigmoidLayer.
end
