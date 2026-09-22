0. research文件夹负责管理该研究项目。
1. 调用子agent去完成调研/coding/验收等子任务，子agent选择opus 5.5 extra high。主agent负责掌管整个项目的进度把控和子agent的编排。
2. 上下文接近上限时会自动压缩，所以不要因为担心 token 用完而提前收尾。如果主agent的context length要满了，则更新research/progress.md，记录时间、现状、当前结论、关注的问题以及未来计划。
3. 在完成原型之后，一次只做一个功能，不要试图一次性完成整个项目。写代码前，先写出该功能的"完成标准"：一份可测试的行为清单。每次完成功能且验收通过之后，更新progress.md并进行git的commit和push。
4. 写通用的实现，不要针对测试用例硬编码。遇到问题，修复根因，不要压制报错。
5. 探索的过程中保持怀疑，保持谦逊，勇于发散思路，积极探索。无效的尝试之后回退即可，并记录到progress.md的对应条目下。
6. research/proposal.md 和 research/plan.md 都是可以在探索的过程中修改的。每一次对proposal和plan的修改都需要单独进行git提交，并保存旧版本+时间戳到research/archive中。定期需要回顾过往proposal和plan确保没有偏离原始目标太远。
7. launch gpu负载之前查询gpu使用状况。每当查询到GPU如果被占用了，挂起等待30分钟再查询。