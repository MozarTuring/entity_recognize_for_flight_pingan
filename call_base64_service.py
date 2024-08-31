import requests
import os
import argparse
import base64
import json
import ipdb
import pickle


parser = argparse.ArgumentParser()
parser.add_argument('--task_name', default='', type=str)
parser.add_argument('--test_dir', required=True, type=str)
parser.add_argument('--get_base64', action='store_true')
parser.add_argument('--save_path', required=True, type=str)
args = parser.parse_args()

# model_url = "http://30.79.106.47:55660/base64"
model_url = "http://boardcards-xingchengdan-gpu-stg.shc-sf-caas.paic.com.cn/base64"

# with open('temp_base64.txt', 'r', encoding='utf8') as wf:
#     base64_data = wf.read()
# data_sent2model = {'base64': base64_data, 'task_name': args.task_name}
# response = requests.post(model_url, json.dumps(data_sent2model), headers={"Content-type": "application/json"})
# r_dict = json.loads(response.text)
# print(r_dict)
dic = {"xingchengdan_3code":[
    '906632dddedfeab4c0fa61f2d885a30ff08', '906640068e3a0874b1b8f3c5fb89b90fd71', '906641503d3b29f4840ba619952042b6082', '906641dd7f1e47b46f29b77392010f206f0', '906647c73b6c0094b52aef4fcb4bc0883ef', '90664e05c45bf094050970b8467b87994a5', '90664e371e567a840a9bbf0b43d04473462', '9066604f9aa24bf40879349974eeb40b1a2', '906669f24a7e5604a1eb08ace2e50c49bd0', '90666b80e14f79a4977929f04f5293ab79d', '9066736519c048c4ded97284e5bff182de9', '9066809841b7266464cb66521a7982188fc', '9066810c097507c47e8aca9f24d6dbb6f0c', '906684d8f57ac7d4802bb46cb52844e6d81', '906688d8cdb06244f97bc320f41cd40b925', '90668cba528ac2c492b88505b3abfe5f6df', '90668d3c63eecd04baea3cad988bd48f374', '9066931a4288345462eb5c6d71df2ccc8c4', '9066951d293af1440f98d8c6e5893d01bed', '90669a08ddc2e3d4a208ac4e2893110c58b', '90669c081210496486b98ac22702ff81627', '90669ef150bc0b7490b92d6f681533aca08', '9066a0fdf6ed57e484eab0578568064fc7c', '9066b494b69cff14415b9db7a6616cea208', '9066ba9d53f0a3b4a1ebf9c4a2872433c16', '9066bb3ee3af36a4861872908c66a9f17f5', '9066c055c4c208547d680d811d5a4f5f7bf', '9066c2d85c19da1452eaacd9ba7260b9aec', '9066c4cd6d303e9469da1a0796f1f680cd3', '9066cebf10219264408b3d8588dd1266693', '9066dd319f65ab94189b13594eeb641d6c5', '9066fd1276ed9c244d9856d9e7173e7751e', '906703bf9cb25ac4ef3b096618f3aa454fe', '90670f9e892b8dd4dcb8419fb9f8f19a308', '906717dacf36ce64517851e44ccedaaa04e', '906719edf75551f422384ad64d0260adf71', '90672a479268937452c81ee32ec7976f7aa', '90672da47e5eb4a425e98b2aae69356465b', '906732c91cd98704a0297f8d98ce5bdca9a', '90673bb59da2f4c430b8655d470c179c034', '90673d3c1aa87574e8883c3494f3e6155d7', '90673f685497e1b46a6ba0192614e7c6aee', '9067445fbcfb5494361a3d03c8181e240f9', '906745ff85b9c48456bbd0a1f7a787e574b', '906747ff49e817f4f689848007647679e73', '906759a4a7e6a274fef8bae826462148f5d', '90675b7215a313f4f0aba018a82f408c826', '906760f6682b9094ff7a63041c6fd95f8ce', '906762c30872f3c4804babeed2cb593d506', '9067638f4b762aa410c894963e60be7dca9', '906770de42c966444d08f3728daaa621b88', '9067716d8ca053a4bada06e462991b2d719', '9067a6f5e754126435c8d78325e48b199bc', '9067d1cdd904c0e42358cd2a1f41d1dbf29', '9067d702df97abf46909fdc35df026a89d1', '9067d8f4d8dd6b54898b18f2cf9bd44e12f', '9067da84e0f1d194bbca78dfcdd72eadad2', '9067deb10ce054d42a78eb61a3a034241d1', '9067df1fc867f9a4198b3b78d0002a1e3b2', '9067e33eecf086748a7b4108e878f8e06b4', '9067e7d5f266d4a4ee9b12d3d72fb027a23', '9067e7d8853da7646c9b9bf39e5673bed5f', '9067e8a8b7d60f2490bab51d3aec5c2c5dc', '9067eaf342a872e43bc9e64f5fc447a187e', '9067ef630fdce5d46dabb6d7b226bd00ab3', '9067f294ae6426646b3817bf85aeb4fd2d3', '90680cea68f676c48df8113b8dec65ceff9', '90680fdb50fdf77475ca175228e261b5d64', '9068227fc4cb9e64a4dae117276a6f408e7', '90682d17b85bdaf4e6086d58d4a9e51544f', '90682e09d9eddc5438cb1fdb7a875177ada', '906832d93162e414b119cb9f76a283788ce', '9068404672fd7c346ada2ef3a31086b59f3', '90684a4539b24cd44799fd98a3bb079f636', '90684b75e7a952944cbafb2d56c2f934c99', '90684c7824e4c7f4daeb0a35f6dd3ebe303', '906850bad89b9c348d9bc67d1fe18923c08', '90685814593bcbd4789a477a818ce9c0267', '90685e4adef36df4e2496e206fe7f34c506', '9068640afc543fd4078910a2e308fa4bbba', '90686a56a1411b94183bb6cdc93d7ddbfb5', '906879d169ed730409998dd6d949d7e9f80', '90687ce4f98368d4e99aeea7f50a5dad293', '906881b347f6e6c41e19c134cd1e4449255', '906899b7eb399a644dd84c6a391c3363b4d', '90689cbce260062455faa34395eb3752620', '9068a2ba607cbc5473db5aecef79a52eade', '9068a6290870e854d239ec9a18f071227b8', '9068a9eb09324b649399ea226c873ad8e41', '9068afd83b8e08644768d174cea665e4ec4', '9068e8884129c57406f80e672814a0c5e0a', '9068eb842c46a664f79a33ec0945e218752', '9068ed1e9ec6f444e16886d6404d6e0bcc7', '9068ee53a9cd1104c979c6be7d0151a1585', '9068f1ed45ab7f54ab489f48f0845f26e15', '9068f5a8879d90d488eabfec53fd1aeb72d', '9068f7272ee014049a49b5f6a3a528467af', '906905dedb6c5644647bf90190667abd0e1', '9069069459163c94096bd6e50bda526ee92', '906908487105e5743a68b4e4f5b2998b759', '90690bcbf5e3c19458082979b78a900d86d', '90690fcd71e96c143ecadc4862eb0c8182e', '90691b8a677473c4501b8b99c0930fbb765', '90691c3ddaa80b94dadb47e3c1ab2d62d19', '90691d08937e23148e9ad496607dabe934d', '9069298d0e600c242c282e50f2d7721727c', '90692fc16736a354b788f9e92f64db30139', '906937ed14c5fc6454b8a46cee9dd5fb3c5', '90693f5e915c961491491df2a13eb016c49', '90694f2cb65c939417eae73690f094b02d8', '9069534d7aa074a4609bcbf003209e8c431', '906965efd104c0f47bea4a52646167de4b4', '90696d81b10f28347d0881d488688321ec6', '90697fad3aa778c474c88a2b70a78ec2e9f', '906984dc5b13d4b4a16a2868e399b89ddd8', '90698fbea542afb4e0984775959150d7a34', '9069911229cd29c469ab55138c12134a5e9', '90699f3618bcc0944e18fdf18595a646533', '9069a4da1ab397c48dd9da09e0c811619a1', '9069a8a6aead3e14ae998b002bc67140142', '9069a98d4ed3cc54068957e9b9fa42478b2', '9069caf6ed309d949ed9331333f58c39218', '9069cbe97ec5e9d46c0a1af6025db2a8df6', '906a0479ab3390240b4b8eb269907866046', '906a0a3268391ba4cac87352895f1dbdc03', '906a1536ebc0d9e4d8e95454a06eda5d77e', '906a1a7e5d245d54487ace691171fda8819', '906a1c0ece505ff44daad1d7df61593b7a8', '906a1f2b33742ba4562b15d4b4d6bffbaa8', '906a20a8ed769d5492091b257c356d684b4', '906a242945dbb454e58a1bb5e40d6c0e3d6', '906a2476e203c1d4267b62637ea4851f038', '906a24e06325c3e490e8c6360a2e6b81c2b', '906a3b86c45f7a848b486ecd263f6bda6a2', '906a4bbf778289d4e1a8961c06803d0f165', '906a4bd983be97e4a119aa368096be3ba4e', '906a557d949c42d446181a697f76fbc3a49', '906a56b7956998b481fba5298b2f077fc0f', '906a58d8c826b134e758f1944d9b8419a8f', '906a75137b17f9047abaefaa81a46202cad', '906a79ce8bf98ee40d8a74280a69c6684bb', '906a7b7c9df1223427796a23072342797af', '906a7dd09d9e3c24ca8b6ad7f75372ca240', '906a828ec1b75ef43f6a4618827776bf305', '906a8504968597d4a14b16873693b265674', '906a88a0b7b0a234b1f8a823e2501ae9dff', '906a8ca47a9edac44458c7dd2c9983641bf', '906a8e1c5e31c3a4ddba6a5ca6fc2fbe76e', '906a97db1b70fee44dd8fbf5c4285c32c29', '906a9d06bccd61641b19ea23d947d464bf5', '906aaa1aade7ba246e4ad3eb9cf16c838c5', '906ac38178f1e254f81b8c692fe95adeaa9', '906ad723c414329448c972cef8589a39af2', '906adb9ac7f653545e3b915faad188587d5', '906afce643ba8bb4b23836a2b417a50d0b6', '906b061570b0d71477789fe1917d91aa4df', '906b11a2b7432204f52816cd4cebc73ddc6', '906b17d86e6fdf14b31830d072cea78a705', '906b1c11ca364e64e5e81425c9d3dac3cb6', '906b29740d801e34244b91461a55dd0cce5', '906b2d1c35cba1d4d8095bde17276d26f19', '906b332c523bb324eccbc8b095563b8e5fb', '906b339f8f57dab4e06b10e70992bae8c9d', '906b37c6c966cea41b9a982f186c04841ae', '906b50910e5356d4fbd93463fe5a52f38fd', '906b538beb9db614b3691794de0380db56c', '906b6d3b6eee9b84764a1d784157a72af7d', '906b6df165d7fc74281b34dc15cbaf0bf5d', '906b6f672e748294f0fbca408bf53c0bfa0', '906b855cc14372a4dd7b2cca14278e00786', '906b8755e82183148eaafaed28b6a1e9347', '906b8f0f674da1a40709d64ee41d13230f8', '906b9a3aee997774494b80b851764604a8c', '906b9ebdd0624554ca887a4ee42a0d43fb3', '906bb3a96eff34141b8929e0a7057920cb3', '906bb7f127d15b84e56ad82bfa22d2b6dd8', '906cd90059a6de64ddfb01b885ac43bbc24', '906cfa0552b64e94815ba5985533f079ff9', '906d0defcb6fc1542819e54ad8fec16ccd1', '906d14b813b9a8d4978a62e31ace3f229c5', '906d1c979bb05d84820a4024ebdbab6de21', '906d27eea476c0945c3806fe43c73729658', '906d2fabc4b3a454a8dab0d25870c51d7ef', '906d4a15b5ab2334672b26f304a8539fe59', '906d4f1510910f24e859f6a5c76e23458ca', '906d5797e04818a400c8672c7958285525f', '906d587f5e5bc4c4cfdaa9b059e67debf75', '906d612452a63ac40f7b5633729fa3120dc', '906d699a944fe664cfd9fb98446c3ca46d0', '906d6c20e038e464df6838032e4862822e9', '906d93a3d53e1a84d84b2952885fee74318', '906d98dd7757b1a4779943e2131b7082d5c', '906da0c2058c09e49d7b6c6432b34b5d273', '906da1b0e55c963404b9667269182f40ab6', '906da923e8ec6af49f8a403751dfead9feb', '906dd279d9be6294f8e898dea61b6b4e02c', '906dd98e750781d4aab813e6041690d3e00', '906ddf7ecbc3ee54362b4bb0689ff0350e8', '906df1cbb699a2e496c9ed7f658b134b6d5', '906df1ddb256f8d4a26ac92e55c9bbf3cd8', '906dff0ce8d1b5a47a3b247d1a2d2f66b8c', '906e0c2fd34e9614defb9c312c05c1e0cd5', '906e0c91a8fe0c5408097e45a06fda24147', '906e238c26daa9c4fc2966c93ac1d9ce463', '906e33fa26aab7b42d6a0777e8782c4b761', '906e469f2287bdb4d3a9dd7af8f19af4c56', '906e73c4e0fa8f24184a1ca7ad9f6b2f45b', '906e8c582d1f1e74fefaa6e7c8ed902dc42', '906e8deb15cae404afb84117543401d56a5', '906e91eb9d449f94543bec09f14fa864d89', '906e99606b343574f84ac38ab6a51e5dc48', '906ed2c2fb47b86405189d9700d7380951f', '906edb59b8866154cd8818d08045cb548d2', '906eea41b521e1d47218b7d62e396ceeb5a', '906ef86e5950a0b49e8bfac9b42f2087400', '906f042259bc5944bbfbed2ded378b3849b', '906f0770fccd83649bc83da571452e91d8a', '906f21492c8971c4320897821a5ed847b28', '906f27c613eb2e844e3bf27418bb99159b8', '906f2ad0f9e034741deaa42c5410c4d48c7', '906f372cba63eaf4b8187196429c62fd8dd', '906f3aa46d9de3d4b1287f57b3d70cf0dda', '906f6e775e58c6f4cc7b15e437adfc52827', '906f7d80b2a9871475f90531369a6761899', '906f82ae7259dca41e1b9ecebdf4b624b0a', '906f8feaac145ae4a6b9efdee38dcddfb6f', '906f97b34b7308c4071b09d322c2e23cb2b', '906f9a3e87aa264426fb54874cfde1c3d89', '906fb8072fd58e647e988d3e5c47eae94e9', '906fb9a6c4aba724485beff88739dff03d2', '906fbe317f64fa1423491749ab7d90a3800', '906fbe9cf5b12eb4c51b1a901c759f28507', '906fc4a762af29d44aa9ac08707817dc5c9', '906ff3ec168b8874ba08e82d2f81080e6e8', '906ff4b279c601a4a27bc990f24e8ca9f8c', '906ffb07c3bc1ab40efb10acfe5398b069a'],
    "boardcards":[
        "9065098835c6a2e4fceabb601c01c93dc4e", "9065346050c4b034070adddae082aa85442", "906509842456cc5418ea4a663052f81285e", "906508859463646445e98eddab3ff6d8149"]}

base64_ls = []
tmp_files = os.listdir(args.test_dir)
tmp_files.remove('label.xlsx')
files = sorted(tmp_files, key=lambda x:int(x.split('.')[0]))
print(files)
for file in files:
    if file == '2.jpg':
        continue
    cur_path = os.path.join(args.test_dir, file)
    # wf_path = os.path.join(args.test_dir, file.split('.')[0]+'.base64')
    if os.path.isfile(cur_path):
        # try:
        with open(cur_path, 'rb') as f:
            base64_data = base64.b64encode(f.read())
        # with open(wf_path, 'w') as wf:
        #     wf.write(base64_data.decode())
        base64_ls.append(base64_data.decode())
        # except:
        #     continue

# for ele in dic[args.task_name]:
#     count = 0
#     for file in os.listdir(args.test_dir):
#         if ele in file:
#             cur_path = os.path.join(args.test_dir, file)
#             # wf_path = os.path.join(args.test_dir, file.split('.')[0]+'.base64')
#             if os.path.isfile(cur_path):
#                 with open(cur_path, 'rb') as f:
#                     base64_data = base64.b64encode(f.read())
#                 # with open(wf_path, 'w') as wf:
#                 #     wf.write(base64_data.decode())
#                 base64_ls.append(base64_data.decode())
#             count += 1
#     if count != 1:
#         print(count, ele)

if args.get_base64:
    exit()

def loader(inp_ls, bs=8):
    i = 0
    while len(inp_ls)-i >= bs:
        yield inp_ls[i:i+bs]
        i += bs
    if len(inp_ls)-i > 0:
        yield inp_ls[i:]

data_rets = []
cur_num = 0
for batch_base64 in loader(base64_ls, 1):
    data_sent2model = {'base64': batch_base64, 'task_name': args.task_name}
    # ipdb.set_trace()
    response = requests.post(model_url, json.dumps(data_sent2model), headers={"Content-type": "application/json"})
    data_rets.extend(json.loads(response.text))
    cur_num += len(batch_base64)
    print("Complete {}".format(cur_num))

print(data_rets)
# entity_lls = []
# catego_ls = ['Name', 'Flight_Number', 'Flight_Date', 'Origin', 'Destination', 'Ticket_Fee', 'Develop_Fee', 'Oil_Fee', 'Other_Fee', 'Total_Fee', "Origin_Code", "Destination_Code"]

with open(args.save_path+'.pkl', 'wb') as wf:
    pickle.dump(data_rets, wf)

# ipdb.set_trace()
# for data_ret in data_rets:
#     entity_ls = []
#     for catego in catego_ls:
#         if catego in data_ret['en_result']:
#             cur_res = data_ret['en_result'][catego].split(' ')
#             entity_ls.extend([catego+'+=+'+ele for ele in cur_res])
#     entity_lls.append(entity_ls)

# with open('entity_lls_0426.pkl', 'wb') as wf:
#     pickle.dump(entity_lls, wf)

# ipdb.set_trace()
print('end')
