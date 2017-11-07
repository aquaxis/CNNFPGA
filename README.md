ソフトウェア技術者のためのFPGA入門 機械学習編 ワークフォルダ

本ワーキングフォルダはSDSoC 2017.2で使用できるワーキングフォルダとなっています。

次のようにリポジトリをクローン又はダウンロードしてください。

```text
git clone git://github.com/CNNFPGA
```

SDSoCを起動します。

SDSoC起動時にワーキングスペースをクローンしたフォルダにします。

 * 起動時に「Select a directory as workspace」ダイアログが表示される場合はここの「Workspace」に下記のフォルダを指定します。
 * 起動時に「Select a directory as workspace」ダイアログが表示さずにSDSoCが起動した場合は、[File]-[Switch Workspace]-[Other...]を選択し、下記のようにワークスペースを指定します。

```text
/home/hoge/CNNFPGA
```

最初はSDSoCの「Project Explorer」には何もプロジェクトが表示されていませので[File]-[Open Projects from File Systems...]を選択し、「Import Projects from File System or Archive」ダイアログで「Import source」を上記のフォルダに指定します。

フォルダの指定が行えると「Folder」タブにCNN_Test0〜40までのフォルダ一覧が表示されるのでチェックされていることを確認して「Finish」ボタンをクリックします。

指定したフォルダのインポートが完了すると「Project Explorer」にCNN_Test0〜40までのプロジェクトが表示されます。

## SDSoCプロジェクト概要

 * ケース０：ソフトウェアのみのSDSoCプロジェクト
 * ケース１：CalcConvolution関数をFPGA化するSDSoCプロジェクト（mem_attributeプラグマの適用）
 * ケース２：CalcConvolution関数をFPGA化するSDSoCプロジェクト（access_patternプラグマの適用）
 * ケース３：Convolution関数をFPGA化するSDSoCプロジェクト
 * ケース４：Convolution関数及びPooling関数をFPGA化するSDSoCプロジェクト
 * ケース５：CNNLayer関数をFPGA化するSDSoCプロジェクト

 * ケース１０：Convolution関数をFPGA化し、トレース機能の追加したSDSoCプロジェクト
 * ケース１１：Convolution関数のデータ用バッファをmemcpyで一旦、コピーする改修をしてFPGA化し、トレース機能の追加したSDSoCプロジェクト
 * ケース１２：ケース１２をCNNLayer関数に変更してFPGA化し、トレース機能の追加したSDSoCプロジェクト
 * ケース１３：ケース１３のCNNLayer関数のデータ用バッファを移動してFPGA化し、トレース機能の追加したSDSoCプロジェクト

 * ケース２０：CNNLayer関数にPIPELINEプラグマを適用したSDSoCプロジェクト
 * ケース２１：CNNLayer関数にUNROLLプラグマを適用したSDSoCプロジェクト
 * ケース２２：CNNLayerの全展開
 * ケース２３：CNNLayerの全展開＋PIPELINEプラグマ

 * ケース３０：CNN処理全体(CNN3レイヤ+全結合)をFPGA化したSDSoCプロジェクト
 * ケース３１：ケース１３を元にCNN処理全体(CNN3レイヤ+全結合)をFPGA化したSDSoCプロジェクト
 * ケース３２：ケース５を元にSDSoCプロジェクト
 * ケース３３：CNNLayer0のみケース１２を適用したSDSoCプロジェクト
 * ケース３４：CNNLayerを使用したSDSoCプロジェクト

 * ケース４０：Convolution関数をエミュレータで確認するSDSoCプロジェクト

 * ケース５０：ケース３０をZCU102のSDSoCプロジェクト

## ケース０：ソフトウェアのみのSDSoCプロジェクト

本プロジェクトは関数のFPGA化は行わず、開発したソフトウェアが正常にコンパイルを通せるのかを確認するためのプロジェクトです。
本プロジェクトのソースコードはPCで開発されたソースコードをSDSoCへ適用する第一歩のプロジェクトの位置付けです。

## ケース１：CalcConvolution関数をFPGA化するSDSoCプロジェクト（mem_attributeプラグマの適用）

本プロジェクトはCalcConvolution関数をFPGA化します。
CalcConvolution関数は本アプリケーションの再下位層の関数です。
最下位層から関数のFPGA化が行えるか試みるプロジェクトです。

本関数でデータ転送を行うfiltter、input_dataにmem_attributeプラグマを適用してFPGA化を試みます。

## ケース２：CalcConvolution関数をFPGA化するSDSoCプロジェクト（access_patternプラグマの適用）

本プロジェクトはCalcConvolution関数をFPGA化します。
ケース１と違う点は本関数でデータ転送を行うfiltter、input_dataに適用するプラグマをaccess_patternに変更した点です。

## ケース３：Convolution関数をFPGA化するSDSoCプロジェクト

本プロジェクトはConvolution関数をFPGA化します。
本関数はCalcConvolution関数の上位関数であり、本関数がFPGA化が可能なのか試みます。

## ケース４：Convolution関数及びPooling関数をFPGA化するSDSoCプロジェクト

本プロジェクトはConvolution関数とPooling関数の2関数をFPGA化します。
本プロジェクトは複数関数のFPGA化を試みます。

## ケース５：CNNLayer関数をFPGA化するSDSoCプロジェクト

本プロジェクトはCNNLayer関数をFPGA化するプロジェクトです。
本関数はConvolution関数及びPooling関数の上位関数になります。

本関数のFPGA化によって効果があるのか確認するプロジェクトです。

## ケース１０：Convolution関数をFPGA化し、トレース機能の追加したSDSoCプロジェクト

本プロジェクトはConvolution関数をFPGA化し、本関数へのアクセスタイミングをトレースし確認するプロジェクトです。

## ケース１１：Convolution関数のデータ用バッファをmemcpyで一旦、コピーする改修をしてFPGA化し、トレース機能の追加したSDSoCプロジェクト

本プロジェクトはケース１０で得られたトレース結果から、Convolution関数の配列を関数内で一時バッファリングするように改修し、その効果があることを確認するプロジェクトです。

## ケース１２：CNNLayer関数のデータ用バッファをmemcpyで一旦、コピーする改修をしてFPGA化し、トレース機能の追加したSDSoCプロジェクト

本プロジェクトではケース１１で得られた結果から、Convolution関数の上位かんすであるCNNLayer関数について、配列を一時バッファリングするように改修し、その効果があることを確認するプロジェクトです。

## ケース２０：CNNLayer関数にPIPELINEプラグマを適用したSDSoCプロジェクト

本プロジェクトではPILELINEプラグマを指定して効果があるか確認するプロジェクトです。

## ケース２１：CNNLayer関数にUNROLLプラグマを適用したSDSoCプロジェクト

本プロジェクトではUNROLLプラグマを指定して効果があるか確認するプロジェクトです。

## ケース３０：CNN処理全体(CNN3レイヤ+全結合)をFPGA化したSDSoCプロジェクト

## ケース４０：Convolution関数をエミュレータで確認するSDSoCプロジェクト

